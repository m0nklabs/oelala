### Changed

- The source-media picker is now one tabbed card everywhere it lived inside a settings frame. New
  `SourcePickerCard` component (`Upload File` / `From My Creations` tabs, optional header action and
  badges) mirrors Image-to-Video's already-good "Upload Photo" card, and `CreationsPickerModal`
  gained an `embedded` mode that renders just the media grid so it can sit in a tab instead of
  opening over the settings.
- Migrated: Image to Image, Reframe, Image to Text (button + expanding panel inside the settings
  card → tabbed card) and Video to Text (a "From My Creations" tab in its existing Upload/YouTube
  tab bar).
