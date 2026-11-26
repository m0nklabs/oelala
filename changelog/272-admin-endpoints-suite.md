### Added

- Behavioral suite for the admin endpoints (15 tests): credit adjustment validation (zero/over-limit
  amounts and short reasons rejected) plus the success contract (rpc payload carries admin id,
  amount and reason; new_balance returned), tier update (invalid value 400, rpc payload), status and
  suspension toggles (exact rpc payloads), and the users listing paging contract (offset =
  (page-1)*per_page, per_page capped at 100 by validation, tier filter as eq. param), plus admin/check: obeys the gate both with bypass on
  and off (the off case asserts the gate lookup ran for the current user).
