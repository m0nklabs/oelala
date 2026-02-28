### Added
- Age verification modal (`AgeVerificationModal`) shown when enabling NSFW content for the first time
- Checkbox confirmation ("I am 18+") required before NSFW mode is activated
- Verification stored in `localStorage` and cleared on logout
- `ageVerified` state exposed from `NSFWContext` for downstream use
