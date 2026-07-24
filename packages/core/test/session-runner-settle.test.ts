import { expect, test } from "bun:test"
import { LLMEvent } from "@opencode-ai/ai"
import { settleStep } from "@opencode-ai/core/session/runner/llm"
import type { StepRecord } from "@opencode-ai/core/session/runner/publish-llm-event"
import { PermissionV2 } from "@opencode-ai/core/permission"

// settleStep is pure: every case here is plain values in, plain values out.
// No runner, no fake LLM stream, no database.

const call = (input: Partial<StepRecord["calls"][number]> & { id: string }): StepRecord["calls"][number] => ({
  name: "echo",
  called: true,
  settled: false,
  providerExecuted: false,
  ...input,
})

const record = (input?: Partial<StepRecord>): StepRecord => ({
  outputStarted: true,
  providerFailed: false,
  calls: [],
  ...input,
})

const noExits: Parameters<typeof settleStep>[0]["tools"] = {
  interrupted: false,
  declines: [],
  failure: undefined,
  infraError: undefined,
}

const finish = (reason: "stop" | "tool-calls"): NonNullable<StepRecord["finish"]> => ({
  finish: reason,
  tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
})

const clean = {
  record: record(),
  tools: noExits,
  streamSucceeded: true,
  streamInterrupted: false,
  settledFailed: false,
  stepLimitReached: false,
}

test("a settled local call completes the step and requires continuation", () => {
  const settlement = settleStep({
    ...clean,
    record: record({ finish: finish("tool-calls"), calls: [call({ id: "c1", settled: true })] }),
  })
  expect(settlement.closing).toEqual([])
  expect(settlement.end).toMatchObject({ _tag: "Completed", needsContinuation: true })
})

test("the step allowance caps continuation", () => {
  const settlement = settleStep({
    ...clean,
    stepLimitReached: true,
    record: record({ finish: finish("tool-calls"), calls: [call({ id: "c1", settled: true })] }),
  })
  expect(settlement.end).toMatchObject({ _tag: "Completed", needsContinuation: false })
})

test("an unresolved hosted call on a clean stream fails the step as result-missing", () => {
  const settlement = settleStep({
    ...clean,
    record: record({ calls: [call({ id: "hosted-1", providerExecuted: true })] }),
  })
  expect(settlement.closing.map((close) => close._tag)).toEqual(["Sweep", "FailAssistant"])
  expect(settlement.failure).toMatchObject({ type: "tool.result-missing" })
  expect(settlement.end).toMatchObject({ _tag: "Failed" })
})

test("a decline settles its own call with its reason, sweeps the rest, and halts", () => {
  const declined = LLMEvent.toolCall({ id: "c1", name: "write", input: {} })
  const settlement = settleStep({
    ...clean,
    record: record({ calls: [call({ id: "c1" }), call({ id: "c2", called: false })] }),
    tools: { ...noExits, declines: [{ call: declined, reason: new PermissionV2.DeclinedError() }] },
  })
  expect(settlement.closing).toMatchObject([
    { _tag: "Decline", callID: "c1", error: { message: "The user declined this tool call" } },
    { _tag: "Sweep", scope: "all" },
    { _tag: "FailAssistant", error: { message: "Step interrupted" } },
  ])
  expect(settlement.end._tag).toBe("HaltDeclined")
})

test("a provider failure sweeps unsettled calls and preserves the recorded failure", () => {
  const settlement = settleStep({
    ...clean,
    streamSucceeded: false,
    record: record({
      providerFailed: true,
      failure: { type: "aborted", message: "provider exploded" },
      calls: [call({ id: "c1" })],
    }),
  })
  expect(settlement.closing).toMatchObject([{ _tag: "Sweep", scope: "all" }])
  expect(settlement.failure).toMatchObject({ message: "provider exploded" })
  expect(settlement.end._tag).toBe("HaltStream")
})
