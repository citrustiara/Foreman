# Plan — Cost Logic & UI Scrolling

## Status: ✅ Complete

## Problem
1. Cost calculation was using `cost_per_1k` (cost per 1,000 tokens) but the pricing constants in `ModelProfile` were actually per-million-token rates. This caused 12,307 input + 857 output tokens to show as $14.87 instead of ~$0.024.
2. Chat panel did not auto-scroll to bottom when new messages appeared.

## Changes

### Cost Fix (per-1K → per-1M)
- `foreman/models/profiles.py`: Renamed `cost_per_1k_input`/`cost_per_1k_output` → `cost_per_1m_input`/`cost_per_1m_output` on `ModelProfile` dataclass and all presets.
- `foreman/tokens/cost.py`: Updated `SessionCost.record()` to accept `cost_per_1m_*` params and divide by 1,000,000 instead of 1,000.
- `foreman/tui/workers.py`: Updated `run_chat` cost recording call to use `cost_per_1m_*` fields.
- `foreman/models/fetcher.py`: Renamed OpenRouter model dict keys to `cost_per_1m_*`.
- `foreman/tui/app.py`: Updated `action_model_selector` model dict keys to `cost_per_1m_*` with `.get()` fallback for older cached data.
- `foreman/tui/widgets.py`: Updated `ModelSelectorScreen._render_table` display to use `cost_per_1m_*`.

### Chat Scrolling
- `foreman/tui/widgets.py`: Added `self.scroll_end()` to `add_user_message`, `add_assistant_message`, `add_system_message`, `add_compact_event`, and `add_error` in `ChatPanel`.
- `foreman/tui/widgets.py`: Relaxed cost display condition from `cost.total_cost > 0` to `cost.total_cost > 0 or cost.entries` so cost panel shows even tiny amounts.

### Verification
- 12,307 input / 857 output tokens with Gemini 2.5 Pro now calculates to **$0.024** (correct).
- Previously: **$14.87** (wrong — was using per-1K math with per-1M constants).
