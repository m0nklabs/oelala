### Changed

- The procedure allowed "at most three merges per run"; the global rule allows **one deliberate
  merge per pass**. Corrected, and made the reason explicit: a run that lands several pull requests
  cannot give each of them the activity read, the base check and the merged-tree test, so it is a
  sweep with a merge button rather than a merge decision. With more than one eligible PR the run
  merges the oldest and leaves the rest to their own clocks.
