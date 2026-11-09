### Changed

- The merge policy moved from **two days** of mergeability to **24 hours**, matching the revised
  user-level rule. The clock starts when the last outstanding review is answered — or, with
  nothing to review, at the last push on the head — and operator silence for that window is
  standing permission.
- Reading the pull request's latest activity is now part of the merge, not a formality: a reply can
  carry a question, a suggestion or context, and a merge is a deliberate action on one pull request
  rather than a side effect of a sweep.
- The closing comment marks the point at which a pull request became merge-ready; it is no longer a
  precondition for merging when there was nothing to review.
