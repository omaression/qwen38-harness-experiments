# VII. Conclusion and Future Work

This report presented a pure-stdlib, cassette-replayed CI regression harness across four LLM features (`bugfix`, `judge`, `review`, `patch`), combining execution-grounded value sandboxing, 12-rule SAST, safe pure-Python `--recount` diff application, cluster-bootstrap/McNemar inference, non-inferiority gating, and a hash-chained promotion ledger. Empirical evaluation across Experiments A–D and three multi-agent ablations yields four actionable engineering rules:

1. **Calibrate gates against measured serving noise and retire ceiling suites.** Small or ceiling-compressed suites (`demo`, `humanevalfix`) cannot separate modest model changes from bf16 serving jitter ($\pm 0.041$ in C0), whereas non-ceiling repository suites (`swe_review_bench`, `swe_coder_effort`) detect genuine capability ($+0.188$ BA, $p=2.26\times 10^{-13}$) and effort gains ($+0.391$ held-out usable rate, $p=5.25\times 10^{-16}$).
2. **Decompose format recovery from underlying judgment.** Always pair uncapped comparisons with `both_valid_compare` and capped-base controls so $+0.048$ format improvements never conceal $-0.040$ capability regressions.
3. **Combine reasoning-effort tuning with thinking-length calibration.** Serving coders at `medium` effort eliminates length truncation and cuts tokens $5.02\times$ without hurting usability vs. `low`, while `w700` SFT (as in Experiment B) is required to satisfy strict thinking-window gates without hard serving caps.
4. **Gate agentic acceptance deterministically.** Replacing LLM auditor vetoes with `tests+SAST` plus thinking-window calibration eliminates 34.8% false rejects, lifts resolved tasks $60\% \to 68\%$ with zero false accepts, and saves $7.0\times$ completion tokens per resolved task.

Future work extends the harness with built-in multi-sample no-op calibration, native token-cost gates alongside accuracy, and isolated per-task container execution directly under `llm-eval`.
