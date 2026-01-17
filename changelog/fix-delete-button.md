### Fixed

- **MyMediaTool delete button not working** - Fixed source routing in delete handler:
  - Items with `source: 'user'` now correctly use `/user/media/` endpoint
  - Items with `source: 'comfyui-local'` or `source: 'generated'` use legacy delete
  - Previously all items went to wrong endpoint due to `source === 'storage'` check

- **Batch download not working** - Fixed `handleBatchDownload` to use indices instead of filenames:
  - `selectedItems` Set contains indices, not filenames
  - Now correctly maps indices to items from `sortedMediaList`
