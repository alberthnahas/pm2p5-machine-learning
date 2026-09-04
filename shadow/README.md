# Daily shadow workflow

This directory is the non-public prospective evidence store for the fixed
PM₂.₅ deployment model. The daily job:

1. saves an immutable snapshot of the official BMKG CEWS dashboard;
2. converts station-local observation times to UTC and retains the first value
   observed for each station-hour while preserving later raw snapshots;
3. requests the current 00 UTC CAMS forecast directly from the Copernicus
   Atmosphere Data Store and samples it at all 27 stations;
4. freezes issue-time predictors, runs the versioned deployment bundle, and
labels rows generated after their target time;
5. matches earlier forecasts with observations only after those observations
   appear and refreshes a lead-specific scorecard.

Forecasts, inputs, raw snapshots, and verification rows are retained
indefinitely during the initial 60–90-day evaluation. There is no public upload
and no automatic model retraining. A future model update must be separately
versioned and evaluated so prospective evidence is not erased by adaptive
refitting.

The local cron wrapper writes ordinary process output to `logs/cron.log` and
structured run records to `logs/runs.jsonl`. `state/latest_run.json` is the
machine-readable health record. Missing CAMS produces an explicitly degraded
observation-only forecast; schema or feature-construction failure exits
non-zero.

Manual run:

```bash
/usr/bin/flock -n /tmp/aq-pm25-ml-shadow.lock scripts/run_shadow_daily.sh
```

The installed schedule is 17:15 WIB (10:15 UTC). This reflects direct CAMS
availability rather than pretending model initialization equals product
availability. Under this schedule, +3 h and +6 h targets have already occurred
and are retained only as latency diagnostics; prospective skill claims use
rows whose target time is later than the recorded generation time. The +12 h
lead is prospective only when acquisition completes before 12 UTC.

Observation freshness is evaluated twice: relative to the historical 00 UTC
feature time and relative to the actual generation time. This prevents a
nominally fresh 00 UTC observation from being presented as fresh when the
forecast is not produced until many hours later.

Resource budget: 4–8 CPU cores, 4 GiB RAM, and 2 GiB working space. Normal
runtime should be 5–30 minutes, dominated by the external CAMS request; model
inference itself is seconds.
