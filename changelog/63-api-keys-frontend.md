### Added

- **API Keys Management UI**: New "API Keys" tool in the Account section of the dashboard
- Create new API keys with custom names and expiration periods
- View all API keys with usage statistics (usage count, last used date)
- Enable/disable API keys without deleting them
- Delete (revoke) API keys permanently
- Copy newly created API key to clipboard (shown only once!)
- Usage instructions with example curl command

### Frontend Components

- `APIKeysTool.jsx`: Full-featured API key management interface
- Added to navigation under new "Account" section
- Responsive modals for create/delete confirmations
