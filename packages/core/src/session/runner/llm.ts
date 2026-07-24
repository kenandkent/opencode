export * as SessionRunnerLLM from "./llm"

import { LLMClient, LLMError, LLMEvent, isContextOverflowFailure, type ProviderErrorEvent, type ToolCall } from "@opencode-ai/ai"
import { Cause, Data, Effect, Exit, Fiber, FiberSet, Layer, Option, Pull, Schedule, Semaphore, Stream } from "effect"
import { Database } from "../../database/database"
import { EventV2 } from "../../event"
import { PermissionV2 } from "../../permission"
import { QuestionTool } from "../../tool/question"
import { ToolOutputStore } from "../../tool-output-store"
import { InstructionState } from "../instruction-state"
import { SessionCompaction } from "../compaction"
import { SessionContext } from "../context"
import { SessionEvent } from "../event"
import { SessionPending } from "../pending"
import { SessionModelRequest } from "../model-request"
import { SessionMessage } from "../message"
import { SessionSchema } from "../schema"
import { SessionStore } from "../store"
import { SessionTitle } from "../title"
import { Service } from "./index"
import { createLLMEventPublisher, type StepRecord } from "./publish-llm-event"
import { SessionError } from "@opencode-ai/schema/session-error"
import { Snapshot } from "../../snapshot"
import { makeLocationNode } from "@opencode-ai/util/effect/app-node"
import { llmClient } from "../../effect/app-node-platform"
import { StepFailedError } from "../error"
import { toSessionError } from "../to-session-error"
import { SessionRunnerRetry } from "./retry"
import { SessionUsage } from "../usage"

/** How one model call ended: settled, awaiting a scheduled retry, or restarted by compaction. */
type CallOutcome = Data.TaggedEnum<{
  Completed: { readonly needsContinuation: boolean; readonly step: number }
  Retry: { readonly step: number }
  Restart: { readonly step: number; readonly recoveredOverflow: boolean }
}>
const CallOutcome = Data.taggedEnum<CallOutcome>()

// Declining an interactive prompt halts the drain instead of becoming model-facing tool output.
const isDecline = (
  error: SessionModelRequest.ExecuteError,
): error is PermissionV2.DeclinedError | QuestionTool.CancelledError =>
  error._tag === "PermissionV2.DeclinedError" || error._tag === "QuestionTool.CancelledError"

/**
 * Classifies how the owned tool fibers ended. Interrupts abort the step; a user decline
 * settles its own call and then aborts the step; a defect from a tool implementation
 * becomes a failed tool call the model can read; a typed infrastructure failure must
 * fail the assistant and then the drain.
 */
const classifyToolExits = (
  settled: Exit.Exit<Array<Exit.Exit<void, SessionModelRequest.ExecuteError>>, never>,
  calls: ReadonlyArray<ToolCall>,
) => {
  // Exits align with calls by construction: one owned fiber per accepted local call.
  const exits = settled._tag === "Success" ? settled.value : []
  const declines = exits.flatMap((exit, index) =>
    exit._tag === "Failure"
      ? exit.cause.reasons.flatMap((reason) =>
          Cause.isFailReason(reason) && isDecline(reason.error) ? [{ call: calls[index], reason: reason.error }] : [],
        )
      : [],
  )
  const causes =
    settled._tag === "Failure" ? [settled.cause] : exits.flatMap((exit) => (exit._tag === "Failure" ? [exit.cause] : []))
  // The first non-interrupt, non-decline failure, rebuilt without decline reasons so the
  // drain's error channel never carries a decline.
  const failure = causes.flatMap((cause) => {
    if (Cause.hasInterrupts(cause)) return []
    const reasons = cause.reasons.flatMap((reason): Array<Cause.Reason<ToolOutputStore.Error>> =>
      Cause.isFailReason(reason) ? (isDecline(reason.error) ? [] : [Cause.makeFailReason(reason.error)]) : [reason],
    )
    return reasons.length > 0 ? [Cause.fromReasons(reasons)] : []
  }).at(0)
  return {
    interrupted: causes.some(Cause.hasInterrupts),
    declines,
    failure,
    infraError: failure === undefined ? undefined : Option.getOrUndefined(Cause.findErrorOption(failure)),
  }
}

/** One durable closing action the settlement issues through the step's writer. */
type Close = Data.TaggedEnum<{
  Decline: { readonly callID: string; readonly error: SessionError.Error }
  Sweep: { readonly error: SessionError.Error; readonly scope: "all" | "hosted" }
  FailAssistant: { readonly error: SessionError.Error }
}>
const Close = Data.taggedEnum<Close>()

/** How callModel resolves once the closing actions are applied. */
type StepEnd = Data.TaggedEnum<{
  HaltStream: {} // rethrow the feed's cause
  HaltDeclined: {} // interrupt: the user declined an interactive prompt
  HaltTools: {} // rethrow the classified tool failure
  HaltSettled: {} // rethrow the settlement await's cause
  Failed: { readonly error: SessionError.Error }
  Completed: { readonly needsContinuation: boolean }
}>
const StepEnd = Data.taggedEnum<StepEnd>()

const TOOLS_INTERRUPTED = { type: "aborted", message: "Tool execution interrupted" } as const
const STEP_INTERRUPTED = { type: "aborted", message: "Step interrupted" } as const
const RESULT_MISSING = { type: "tool.result-missing", message: "Provider did not return a tool result" } as const

/**
 * Pure settlement: given the step record, the classified tool exits, and how the feed
 * ended, decide every closing action and how the step resolves. No IO; the caller applies
 * `closing` in order through the writer, then publishes the terminal event `failure` and
 * `record.finish` select.
 */
export const settleStep = (input: {
  readonly record: StepRecord
  readonly tools: ReturnType<typeof classifyToolExits>
  readonly streamSucceeded: boolean
  readonly streamInterrupted: boolean
  readonly settledFailed: boolean
  /** A thrown LLM failure not already recorded as the step's provider error. */
  readonly llmError?: SessionError.Error
  readonly stepLimitReached: boolean
}) => {
  const { record, tools } = input
  const closing: Array<Close> = []
  let failure = record.failure
  // Mirror the writer's settled marks so each sweep sees the closings before it.
  const settled = new Set(record.calls.filter((call) => call.settled).map((call) => call.id))
  const sweep = (error: SessionError.Error, scope: "all" | "hosted") => {
    const targets = record.calls.filter(
      (call) => !settled.has(call.id) && (scope === "all" || call.providerExecuted),
    )
    for (const call of targets) settled.add(call.id)
    if (targets.length > 0) closing.push(Close.Sweep({ error, scope }))
    return targets.length > 0
  }
  // failAssistant also settles never-called input fragments (the writer's "uncalled" sweep).
  const failAssistant = (error: SessionError.Error) => {
    for (const call of record.calls) if (!call.called) settled.add(call.id)
    closing.push(Close.FailAssistant({ error }))
    failure ??= error
  }

  // A thrown LLM failure fails the assistant before any sweep.
  if (input.llmError) failAssistant(input.llmError)
  // A declined call settles durably with its reason before the generic sweeps.
  for (const decline of tools.declines) {
    settled.add(decline.call.id)
    closing.push(
      Close.Decline({
        callID: decline.call.id,
        error: {
          type: "aborted",
          message:
            decline.reason._tag === "QuestionTool.CancelledError"
              ? decline.reason.message
              : "The user declined this tool call",
        },
      }),
    )
  }
  if (tools.declines.length > 0 || input.streamInterrupted || tools.interrupted) {
    sweep(TOOLS_INTERRUPTED, "all")
    failAssistant(STEP_INTERRUPTED)
  }
  if (tools.failure !== undefined) {
    const error = toSessionError(tools.infraError ?? Cause.squash(tools.failure))
    sweep(error, "all")
    if (tools.infraError !== undefined) failAssistant(error)
  }
  // Fail unresolved calls before the terminal step event. Local calls have joined, so
  // these sweeps only close calls that could not produce a truthful settlement.
  if (record.providerFailed) sweep(TOOLS_INTERRUPTED, "all")
  if (input.llmError) sweep(RESULT_MISSING, "hosted")
  // A clean stream that still left hosted calls unresolved fails the step itself.
  if (input.streamSucceeded && !record.providerFailed) {
    const hostedResultMissing = sweep(RESULT_MISSING, "hosted")
    if (hostedResultMissing && !record.finish) failAssistant(RESULT_MISSING)
  }

  const end = (() => {
    if (!input.streamSucceeded) return StepEnd.HaltStream()
    if (tools.declines.length > 0) return StepEnd.HaltDeclined()
    if ((tools.interrupted || tools.infraError !== undefined) && tools.failure) return StepEnd.HaltTools()
    if (tools.interrupted && input.settledFailed) return StepEnd.HaltSettled()
    if (failure) return StepEnd.Failed({ error: failure })
    return StepEnd.Completed({
      // A local call or malformed tool input requires another model step, unless this
      // step already exhausted the agent's allowance.
      needsContinuation:
        !input.stepLimitReached &&
        record.calls.some((call) => !call.providerExecuted && (call.called || call.settled)),
    })
  })()

  return { closing, failure, end }
}

const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const events = yield* EventV2.Service
    const llm = yield* LLMClient.Service
    const store = yield* SessionStore.Service
    const context = yield* SessionContext.Service
    const modelRequests = yield* SessionModelRequest.Service
    const snapshots = yield* Snapshot.Service
    const db = (yield* Database.Service).db
    const compaction = yield* SessionCompaction.Service
    const title = yield* SessionTitle.Service
    // Title generation is a side effect of the first step; it must not delay step continuation.
    // Tracked per process so repeated wakes before the second user message arrives don't
    // re-fire a redundant LLM call; `SessionTitle` itself is idempotent based on durable history.
    const titleStarted = new Set<SessionSchema.ID>()
    const forkTitle = yield* FiberSet.makeRuntime<never, void, never>()
    /**
     * Drains eligible manual compaction and user input until the Session becomes idle.
     * Execution lifecycle is published per busy period by SessionExecution, not here.
     */
    const drain = Effect.fn("SessionRunner.drain")(function* (input: {
      readonly sessionID: SessionSchema.ID
      readonly force: boolean
    }) {
      if (!input.force && !(yield* SessionPending.has(db, input.sessionID, "any"))) return
      yield* settleStaleToolCalls(input.sessionID)
      yield* runPendingCompaction(input.sessionID)
      if (!input.force && !(yield* SessionPending.has(db, input.sessionID, "input"))) return
      do {
        yield* runSteps(input.sessionID)
      } while (yield* SessionPending.has(db, input.sessionID, "input"))
    })

    /**
     * Runs logical steps until no tool result or newly admitted steer requires another
     * model call. Queued inputs remain pending until the current model work reaches idle.
     */
    const runSteps = Effect.fn("SessionRunner.runSteps")(function* (sessionID: SessionSchema.ID) {
      // Fresh work may promote queued input; later steps absorb steers only.
      let promotable: SessionPending.Promotable = "input"
      let step = 1
      while (true) {
        const result = yield* runStep(sessionID, promotable, step)
        yield* startTitleOnce(sessionID)
        yield* runPendingCompaction(sessionID)
        if (!result.needsContinuation && !(yield* SessionPending.has(db, sessionID, "steer"))) return
        promotable = "steer"
        step = result.step + 1
      }
    })

    /** Completes one logical model step, transparently retrying or rebuilding after compaction. */
    const runStep = Effect.fnUntraced(function* (
      sessionID: SessionSchema.ID,
      promotable: SessionPending.Promotable,
      step: number,
    ) {
      // Minting message identity before any attempt lets retries resume the same durable
      // message. A compaction restart re-mints: the old message is stranded behind the new
      // compaction boundary, so the rebuilt step needs identity inside the new epoch.
      let assistantMessageID = SessionMessage.ID.create()
      const retry = yield* Schedule.toStepWithSleep(
        SessionRunnerRetry.schedule(events, sessionID, () => assistantMessageID),
      )
      /**
       * Consumes one retry allowance: sleeps the scheduled backoff, or publishes
       * Step.Failed and fails once attempts are exhausted. The step loop performs
       * the retry itself on the next iteration.
       */
      const waitForRetry = (failure: SessionRunnerRetry.RetryableFailure) =>
        retry(failure).pipe(
          Effect.as(CallOutcome.Retry({ step: failure.step })),
          Pull.catchDone(() =>
            events
              .publish(SessionEvent.Step.Failed, {
                sessionID,
                assistantMessageID,
                error: failure.error,
              })
              .pipe(Effect.andThen(Effect.fail(failure.cause))),
          ),
        )
      let currentPromotable: SessionPending.Promotable | undefined = promotable
      let currentStep = step
      // Overflow recovery is one-shot: a call after recovery must not recover another overflow.
      let recoverOverflow = true
      while (true) {
        const outcome = yield* callModel(
          sessionID,
          currentPromotable,
          currentStep,
          recoverOverflow,
          assistantMessageID,
        ).pipe(Effect.catchTag("SessionRunner.RetryableFailure", waitForRetry))
        if (outcome._tag === "Completed") return { needsContinuation: outcome.needsContinuation, step: outcome.step }
        if (outcome._tag === "Restart") {
          if (outcome.recoveredOverflow) recoverOverflow = false
          assistantMessageID = SessionMessage.ID.create()
        }
        // Neither a retry nor a compaction restart re-promotes input.
        currentPromotable = undefined
        currentStep = outcome.step
      }
    })

    /**
     * Prepares and runs at most one model call, executes its local tools, and durably
     * settles the step. Compaction may instead request that the logical step restart.
     */
    const callModel = Effect.fn("SessionRunner.callModel")(function* (
      sessionID: SessionSchema.ID,
      promotable: SessionPending.Promotable | undefined,
      step: number,
      recoverOverflow: boolean,
      assistantMessageID: SessionMessage.ID,
    ) {
      const selected = yield* context.select(sessionID)
      // Establish what the model knows before admitting what the user said, so
      // a blocked first step leaves pending inputs untouched.
      yield* InstructionState.prepare(db, events, selected.instructions, selected.session.id)
      const promoted = promotable ? yield* SessionPending.promote(db, events, selected.session.id, promotable) : 0
      // Promoted input opens a fresh step allowance.
      const currentStep = promoted > 0 ? 1 : step
      const loaded = yield* context.load(selected)
      const { session, agent } = loaded
      const resolved = loaded.model
      const model = resolved.model
      // Make room: history must fit the context window before the call. A pending manual
      // compaction owns this instead; the runner executes it between steps.
      const compactionInput = { session, messages: loaded.messages, model, cost: resolved.cost }
      if (compaction.required(compactionInput) && !(yield* SessionPending.compaction(db, session.id))) {
        const compacted = yield* compaction.compact(compactionInput)
        if (compacted.status === "completed")
          return CallOutcome.Restart({ step: currentStep, recoveredOverflow: false })
        return yield* new StepFailedError({ error: compacted.error })
      }
      const prepared = yield* modelRequests.prepare({
        context: loaded,
        step: currentStep,
      })
      // Every local tool call forked here is owned until it reaches one durable settlement.
      const toolRuns: Array<{
        readonly call: ToolCall
        readonly fiber: Fiber.Fiber<void, SessionModelRequest.ExecuteError>
      }> = []
      const interruptTools = Effect.suspend(() => Fiber.interruptAll(toolRuns.map((run) => run.fiber)))
      const startSnapshot = yield* snapshots.capture()
      const publisher = createLLMEventPublisher(events, {
        sessionID: session.id,
        agent: agent.id,
        // The selected catalog identity, not model.id: route-level ids are provider API
        // model ids (for example gpt-5.5-fast resolves to api id gpt-5.5).
        model: resolved.ref,
        providerMetadataKey: model.route.providerMetadataKey ?? model.provider,
        snapshot: startSnapshot,
        assistantMessageID,
      })
      const stepUsage = (finish: NonNullable<StepRecord["finish"]>) => ({
        cost: SessionUsage.calculateCost(resolved.cost, finish.tokens),
        tokens: finish.tokens,
      })

      const captureStepEnd = Effect.fnUntraced(function* () {
        const snapshot = yield* snapshots.capture()
        const files =
          startSnapshot && snapshot
            ? yield* snapshots
                .files({ from: startSnapshot, to: snapshot })
                .pipe(Effect.catch(() => Effect.succeed(undefined)))
            : undefined
        return { snapshot, files }
      })

      const publishStepEnd = (finish: NonNullable<StepRecord["finish"]>) =>
        Effect.gen(function* () {
          const end = yield* captureStepEnd()
          yield* events.publish(SessionEvent.Step.Ended, {
            sessionID: session.id,
            assistantMessageID: yield* publisher.startAssistant(),
            finish: finish.finish,
            ...stepUsage(finish),
            ...end,
          })
        })

      // Only two durable writers can ever collide: the provider loop and a tool fiber
      // publishing its own completion; each takes the permit. Settlement never contends,
      // because it writes only after every owned fiber has exited.
      const publication = Semaphore.makeUnsafe(1)
      const permit = <A, E, R>(effect: Effect.Effect<A, E, R>) => publication.withPermit(effect)

      // The stream is defined here but runs inside the settlement mask below: publish each
      // event durably, fork one fiber per local tool call, and hold back a virgin
      // context-overflow provider error so settlement may recover it via compaction.
      let overflowFailure: ProviderErrorEvent | undefined
      const providerStream = llm.stream(prepared.request).pipe(
        Stream.runForEach((event) =>
          Effect.gen(function* () {
            if (overflowFailure || publisher.hasProviderError()) return
            if (LLMEvent.is.providerError(event)) {
              if (isContextOverflowFailure(event) && !publisher.record().outputStarted) {
                overflowFailure = event
                return
              }
            }
            yield* permit(publisher.publish(event))
            if (event.type !== "tool-call" || event.providerExecuted) return
            const assistantMessageID = yield* publisher.assistantMessageID(event.id)
            toolRuns.push({
              call: event,
              fiber: yield* Effect.uninterruptibleMask((restore) =>
                restore(
                  prepared.executeTool({
                    sessionID: session.id,
                    agent: agent.id,
                    messageID: assistantMessageID,
                    call: event,
                    // Progress is ephemeral, not durable history: nothing to serialize.
                    progress: (update) => publisher.progress(event.id, update),
                  }),
                ).pipe(
                  // The fiber owns its call: it publishes its own completion, masked so a
                  // finished execution always reaches its durable settlement.
                  Effect.flatMap((outcome) => permit(publisher.toolExecution(event.id, event.name, outcome))),
                ),
              ).pipe(Effect.forkScoped),
            })
          }),
        ),
        Effect.ensuring(permit(publisher.flush())),
      )

      // Settle: only the stream and the fiber joins are interruptible (restore); every
      // other line is protected so a started call always reaches one durable outcome.
      return yield* Effect.uninterruptibleMask((restore) =>
        Effect.gen(function* () {
          const stream = yield* restore(providerStream).pipe(Effect.exit)
          const streamFailure = Option.getOrUndefined(Exit.findErrorOption(stream))
          // Note: Exit.hasInterrupts is a type guard whose false branch unsoundly narrows
          // away non-interrupt failures, so both interrupt checks stay Cause-based.
          const streamInterrupted = stream._tag === "Failure" && Cause.hasInterrupts(stream.cause)

          // Settle every owned tool run first: await all exits, not just the first
          // failure. Afterwards no fiber is alive, settlement is the only writer, and
          // the record is final.
          if (streamInterrupted) yield* interruptTools
          const settled = yield* restore(
            Effect.forEach(toolRuns, (run) => Fiber.await(run.fiber), { concurrency: "unbounded" }),
          ).pipe(Effect.exit)
          if (settled._tag === "Failure") yield* interruptTools
          const tools = classifyToolExits(
            settled,
            toolRuns.map((run) => run.call),
          )

          // A context overflow before any assistant output is recoverable: compact and
          // restart the step instead of surfacing the provider error.
          if (
            recoverOverflow &&
            !publisher.record().outputStarted &&
            isContextOverflowFailure(overflowFailure ?? streamFailure) &&
            (yield* restore(compaction.compact(compactionInput))).status === "completed"
          )
            return CallOutcome.Restart({ step: currentStep, recoveredOverflow: true })

          // An unrecovered held-back overflow becomes the step's durable provider error.
          if (overflowFailure) yield* publisher.publish(overflowFailure)
          const llmFailure = streamFailure instanceof LLMError ? streamFailure : undefined
          const llmError =
            llmFailure && !publisher.record().providerFailed ? toSessionError(llmFailure) : undefined
          if (llmFailure && llmError && SessionRunnerRetry.isRetryable(llmFailure) && !publisher.record().outputStarted) {
            // RetryScheduled and Step.Failed fold onto an existing assistant message, so
            // Step.Started must be durable before the failure escapes.
            yield* publisher.startAssistant()
            return yield* new SessionRunnerRetry.RetryableFailure({
              cause: llmFailure,
              error: llmError,
              step: currentStep,
            })
          }

          // Everything durable is recorded; settle the step as a pure decision, then
          // apply its closing actions in order through the writer.
          const record = publisher.record()
          const settlement = settleStep({
            record,
            tools,
            streamSucceeded: stream._tag === "Success",
            streamInterrupted,
            settledFailed: settled._tag === "Failure",
            llmError,
            stepLimitReached: prepared.stepLimitReached,
          })
          for (const close of settlement.closing) {
            switch (close._tag) {
              case "Decline":
                yield* publisher.failTool(close.callID, close.error)
                break
              case "Sweep":
                yield* publisher.failUnsettledTools(close.error, close.scope)
                break
              case "FailAssistant":
                yield* publisher.failAssistant(close.error)
                break
            }
          }

          if (record.finish && !settlement.failure) yield* publishStepEnd(record.finish)
          if (settlement.failure) {
            const end = yield* captureStepEnd()
            yield* publisher.publishStepFailure({
              ...(record.finish ? stepUsage(record.finish) : {}),
              ...end,
            })
          }

          const end = settlement.end
          if (end._tag === "HaltStream" && stream._tag === "Failure") return yield* Effect.failCause(stream.cause)
          if (end._tag === "HaltDeclined") return yield* Effect.interrupt
          if (end._tag === "HaltTools" && tools.failure) return yield* Effect.failCause(tools.failure)
          if (end._tag === "HaltSettled" && settled._tag === "Failure") return yield* Effect.failCause(settled.cause)
          if (end._tag === "Failed") return yield* new StepFailedError({ error: end.error })
          if (end._tag === "Completed")
            return CallOutcome.Completed({ needsContinuation: end.needsContinuation, step: currentStep })
          return yield* Effect.die(new Error(`Step end lost its cause: ${end._tag}`))
        }),
      )
    }, Effect.scoped)

    /** Executes a previously admitted manual compaction request, if one is pending. */
    const runPendingCompaction = Effect.fn("SessionRunner.runPendingCompaction")(function* (
      sessionID: SessionSchema.ID,
    ) {
      const pending = yield* SessionPending.compaction(db, sessionID)
      if (!pending) return
      const session = yield* getSession(sessionID)
      return yield* Effect.uninterruptibleMask((restore) =>
        Effect.gen(function* () {
          const compacted = yield* restore(
            Effect.gen(function* () {
              return yield* compaction.compactManual({
                session,
                messages: yield* store.context(sessionID),
                inputID: pending.id,
              })
            }),
          ).pipe(Effect.exit)
          if (Exit.isSuccess(compacted)) return
          const unsettled = yield* SessionPending.compaction(db, sessionID)
          if (unsettled)
            yield* events.publish(SessionEvent.Compaction.Failed, {
              sessionID,
              reason: "manual",
              error: Cause.hasInterruptsOnly(compacted.cause)
                ? { type: "aborted", message: "Compaction cancelled" }
                : { type: "compaction.failed", message: Cause.pretty(compacted.cause) },
              inputID: unsettled.id,
            })
          return yield* Effect.failCause(compacted.cause)
        }),
      )
    })

    /** Closes stale tool calls left active by an earlier interrupted drain. */
    const settleStaleToolCalls = Effect.fn("SessionRunner.settleStaleToolCalls")(function* (
      sessionID: SessionSchema.ID,
    ) {
      for (const message of yield* store.context(sessionID)) {
        if (message.type !== "assistant") continue
        for (const tool of message.content) {
          if (tool.type !== "tool" || (tool.state.status !== "streaming" && tool.state.status !== "running")) continue
          yield* events.publish(SessionEvent.Tool.Failed, {
            sessionID,
            assistantMessageID: message.id,
            callID: tool.id,
            error: { type: "aborted", message: `Tool execution interrupted: ${tool.name}` },
            executed: tool.executed === true,
          })
        }
      }
    })

    /** Fires title generation once per process after the first step makes a user message visible. */
    const startTitleOnce = Effect.fnUntraced(function* (sessionID: SessionSchema.ID) {
      if (titleStarted.has(sessionID)) return
      titleStarted.add(sessionID)
      forkTitle(title.generateForFirstPrompt(yield* getSession(sessionID)).pipe(Effect.ignore))
    })

    const getSession = Effect.fn("SessionRunner.getSession")(function* (sessionID: SessionSchema.ID) {
      const session = yield* store.get(sessionID)
      if (!session) return yield* Effect.die(new Error(`Session not found: ${sessionID}`))
      return session
    })

    return Service.of({ drain })
  }),
)

export const node = makeLocationNode({
  service: Service,
  layer,
  deps: [
    EventV2.node,
    llmClient,
    SessionContext.node,
    SessionModelRequest.node,
    SessionStore.node,
    SessionCompaction.node,
    SessionTitle.node,
    Snapshot.node,
    Database.node,
  ],
})
