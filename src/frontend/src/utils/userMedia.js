/**
 * Resolve a user-media file without assuming its media-type segment.
 *
 * Profiles and history entries persist only a filename, but the same file can
 * live under different segments: `uploads/` for source frames the user
 * uploaded, `images/` and `videos/` for generated output. Requesting a
 * hardcoded `/user/media/uploads/<name>` therefore 404s whenever the stored
 * name refers to generated media, which surfaced as NoSuchKey noise in the
 * backend log on every gallery view.
 */

export const PROFILE_MEDIA_SEGMENTS = ['uploads', 'images', 'videos']

/**
 * Fetch the first existing media segment for `name`.
 *
 * @param {Function} apiFetchFn - injected fetch (see utils/mediaUtils.js for the
 *   same convention); keeps this unit-testable without a DOM.
 * @param {string} name - bare filename as stored in the profile/history
 * @param {string[]} [segments] - media-type segments to try, in order
 * @returns {Promise<{response: Response, segment: string}|null>} null when no segment has it
 */
export async function resolveUserMedia(
  apiFetchFn,
  name,
  segments = PROFILE_MEDIA_SEGMENTS
) {
  if (!name) return null
  const encoded = encodeURIComponent(name)
  for (const segment of segments) {
    const response = await apiFetchFn(`/user/media/${segment}/${encoded}`)
    if (response?.ok) return { response, segment }
  }
  return null
}
