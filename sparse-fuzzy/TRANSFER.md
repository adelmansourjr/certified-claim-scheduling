# Cross-domain cost transfer to scientific abstracts

The WiCE timing experiment tests compound Wikipedia claims. Test whether exact character scoring also transfers to single scientific claims paired with whole cited abstracts. Use all released SciFact development claim–cited-abstract pairs from the official archive (SHA-256 pinned in code). Character matches use claim text, titles and full abstract text; they do not use rationale locations, evidence labels or verdicts. One such pair forms a single-atom request. The cohort is a supplied-citation workload, and correctness of scientific judgments is outside this transfer test. The archive covers a public dataset previously used by related workspaces but was not used to select this optimization.

Reuse the complete WiCE model, shared-cache behavior, three candidate index policies, fixed CPU environment, strongest development comparator and fastest frozen WiCE candidate. No tuning or selection on SciFact. Pair and group by claims and cited abstracts. Report exact feature and verdict agreement, ratio of summed five-repeat median CPU times with paired grouped bootstrap, and work. This external sensitivity is descriptive and separate from the WiCE pass criterion.

[Dataset paper, Wadden et al.](https://aclanthology.org/2020.emnlp-main.609/) · [Authors' dataset repository](https://github.com/allenai/scifact)
