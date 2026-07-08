# FREUID Challenge 2026 — Public Leaderboard Snapshot

Pulled 2026-07-08 03:37 UTC via `kaggle competitions leaderboard -c
the-freuid-challenge-2026-ijcai-ecai --download`. 228 teams total. Lower score is better
(AuDET). Our team, **hyunsooochung**, sits at **rank 37 / 228** with `0.00744` — this is
`finetune_v0`'s score; `bayar_dinov2_v1` has not been submitted yet (daily quota).

## Top 15

| Rank | Team | Score | Submissions | Last submission |
|---|---|---|---|---|
| 1 | hoppery | 0.00039 | 62 | 2026-07-06 14:30 |
| 2 | Hardik Sharma | 0.00042 | 17 | 2026-07-07 20:06 |
| 3 | seantangth | 0.00053 | 20 | 2026-07-06 07:55 |
| 4 | SKYisSKYisSKY | 0.00055 | 55 | 2026-07-04 08:06 |
| 5 | tianboguangding | 0.00063 | 48 | 2026-07-07 18:13 |
| 6 | The Unreal Team | 0.00071 | 8 | 2026-07-07 11:27 |
| 7 | MojoJojos | 0.00091 | 17 | 2026-06-26 21:08 |
| 8 | Cloudyy | 0.00112 | 25 | 2026-07-01 15:20 |
| 9 | Kajunoa | 0.00112 | 8 | 2026-07-01 15:10 |
| 10 | TranLuan25 | 0.00118 | 20 | 2026-07-07 08:53 |
| 11 | SCLim | 0.00126 | 44 | 2026-06-22 07:26 |
| 12 | Blue | 0.00137 | 47 | 2026-07-06 12:40 |
| 13 | AIOZ-AI | 0.00146 | 58 | 2026-07-08 01:29 |
| 14 | Endy2001 | 0.00150 | 23 | 2026-07-01 15:22 |
| 15 | nadhir hasan | 0.00169 | 17 | 2026-07-08 02:22 |

## Near our position (ranks 28-45)

| Rank | Team | Score | Submissions | Last submission |
|---|---|---|---|---|
| 28 | Kush Patel | 0.00271 | 40 | 2026-07-06 10:51 |
| 29 | Linh Võ | 0.00318 | 60 | 2026-07-07 10:08 |
| 30 | NTTT | 0.00329 | 13 | 2026-06-29 07:06 |
| 31 | GikitaTan | 0.00414 | 43 | 2026-07-07 15:18 |
| 32 | rrishavrraj | 0.00452 | 7 | 2026-07-05 11:26 |
| 33 | test_freuid | 0.00465 | 60 | 2026-06-23 02:54 |
| 34 | bddfh | 0.00467 | 10 | 2026-07-07 17:03 |
| 35 | RayServe | 0.00590 | 15 | 2026-06-29 07:04 |
| 36 | abc | 0.00721 | 55 | 2026-07-07 12:21 |
| **37** | **hyunsooochung (us)** | **0.00744** | **32** | **2026-07-08 01:19** |
| 38 | Quốc Bảo | 0.00800 | 25 | 2026-06-25 09:50 |
| 39 | Wensgytyb | 0.00804 | 20 | 2026-07-08 00:05 |
| 40 | LLL | 0.01002 | 27 | 2026-06-20 00:35 |
| 41 | Đức Hải Hà | 0.01013 | 18 | 2026-07-07 03:57 |
| 42 | abcxyziiiii | 0.01020 | 5 | 2026-06-29 08:17 |
| 43 | vvlqaz | 0.01020 | 5 | 2026-06-29 08:11 |
| 44 | Supr3mum | 0.01034 | 20 | 2026-07-08 00:04 |
| 45 | Артём Свинобоев | 0.01065 | 21 | 2026-07-06 04:37 |

## Notes

- The gap to 1st place (0.00039) is ~19x; the gap to rank 15 (0.00169) is ~4.4x. The scores
  cluster tightly around our rank (rank 36 = 0.00721, rank 38 = 0.00800) — small gains move
  several ranks, consistent with what CLAUDE.md already noted at `finetune_v0`'s original
  submission (then rank 33/212, now rank 37/228 as more teams have joined/improved).
- `bayar_dinov2_v1`'s `probe_v2_AuDET` (0.0368) is a *local* held-out-split number, not directly
  comparable to this table's public-test scores — see `docs/technical_report.md` for why.
