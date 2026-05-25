### Fixed

- return Wan2.2 local T2I as a real queued local ComfyUI job so normal queue tracking and completion handling apply
- pass the authenticated user id into Wan2.2 local T2I generation so finished images are auto-registered into user media
- register queued local V2 jobs in backend active job tracking so the header queue indicator can display them
- add a backend local-job completion poller and completed-result cache so My Media updates no longer depend on a frontend tab continuously polling job status