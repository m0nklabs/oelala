# Gallery System Documentation

## Overview

The Gallery System allows users to publish their AI-generated content to a community gallery where others can discover, view, and interact with their creations.

## Features

### For Content Creators
- **Publish to Gallery**: Share your best creations with the community
- **Content Control**: Choose what to share with title, description, and tags
- **NSFW Tagging**: Mark content appropriately for age-restricted content
- **Unpublish**: Remove content from the gallery at any time
- **Track Engagement**: See views and likes on your published content

### For Viewers
- **Discover Content**: Browse community creations in a responsive grid
- **Filter & Sort**: Filter by media type and sort by newest, most liked, or most viewed
- **SFW Default**: Anonymous users only see SFW content
- **Detailed View**: Click any item to see full details, prompts, and settings
- **Like & Share**: Interact with content you enjoy (login required for likes)
- **Copy Prompts**: Learn from others by copying their prompts

## Database Schema

### `published_media` Table
Stores published media items with metadata, engagement stats, and content flags.

**Key Fields:**
- `id`: UUID primary key
- `user_id`: Reference to auth.users
- `storage_path`: Path to media file in user storage
- `title`: User-provided title (max 100 chars)
- `description`: Optional description (max 500 chars)
- `tags`: Array of searchable tags
- `is_nsfw`: Content rating flag
- `media_type`: video, image, or audio
- `metadata`: JSONB with prompts, settings, model info
- `view_count`: Total views
- `like_count`: Total likes

### `published_media_likes` Table
Tracks user likes on published media (many-to-many relationship).

### Row Level Security (RLS)
- **Public Read**: Anyone can view SFW content
- **Authenticated Read**: Logged-in users can view NSFW content
- **Owner Write**: Users can only publish/unpublish their own content

## API Endpoints

### Publish Media
```http
POST /api/gallery/publish
Authorization: Bearer <token>
Content-Type: application/json

{
  "storage_path": "video/my-video.mp4",
  "title": "My Amazing Creation",
  "description": "Optional description",
  "tags": ["tag1", "tag2"],
  "is_nsfw": false,
  "media_type": "video",
  "metadata": { "prompt": "...", "model": "..." }
}
```

### Unpublish Media
```http
DELETE /api/gallery/{media_id}
Authorization: Bearer <token>
```

### List Gallery Items
```http
GET /api/gallery?media_type=video&sort_by=created_at&page=1&per_page=30
```

**Query Parameters:**
- `media_type`: all, video, image, audio (optional)
- `is_nsfw`: true/false (optional, forced to false for anonymous users)
- `sort_by`: created_at, like_count, view_count
- `order`: asc, desc
- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 30, max: 100)

### Get Media Details
```http
GET /api/gallery/{media_id}
```

Increments view count automatically.

### Toggle Like
```http
POST /api/gallery/{media_id}/like
Authorization: Bearer <token>
```

Returns new like state and count.

### Get User's Published Items
```http
GET /api/gallery/user/{user_id}?page=1&per_page=30
```

Anonymous users only see SFW content.

## Frontend Components

### PublishModal.jsx
Modal dialog for publishing media items with:
- Title input (required)
- Description textarea (optional)
- Tags input (comma-separated)
- NSFW checkbox
- Media preview
- Form validation

### Gallery.jsx
Main gallery page with:
- Responsive grid layout
- Media type filters (all/video/image)
- Sort options (newest/popular/most viewed)
- Infinite scroll pagination
- SFW filtering for anonymous users
- Click to open detail modal

### MediaDetailModal.jsx
Full-screen modal for viewing media details:
- Large media player/viewer
- Title, description, tags
- View and like counts
- Prompt and settings display
- Copy prompt button
- Like button (auth required)
- Share link button

### MyMediaTool.jsx (Modified)
Added publish button next to favorites:
- Green upload icon when published
- Click to open PublishModal
- Badge shows publication status

## Usage Flow

### Publishing Content
1. User creates content (image/video)
2. Content is saved to user's private storage
3. User clicks "Publish" button in My Media
4. PublishModal opens with content preview
5. User enters title, description, tags, and NSFW flag
6. On submit, API creates `published_media` record
7. Media becomes visible in community gallery

### Viewing Gallery
1. User navigates to Gallery from sidebar
2. Gallery fetches items based on filters
3. Anonymous users see only SFW content
4. Click any item to open MediaDetailModal
5. View count increments automatically
6. Logged-in users can like items
7. Users can copy prompts to learn

## Security Considerations

### Content Safety
- **SFW Default**: Anonymous users cannot see NSFW content
- **User-Controlled**: Creators choose NSFW flag
- **Moderation Ready**: Database structure supports future moderation features

### Access Control
- **Publishing**: Requires authentication
- **Viewing SFW**: Public access
- **Viewing NSFW**: Requires authentication
- **Liking**: Requires authentication
- **Unpublishing**: Owner only (enforced by RLS)

### Privacy
- **User Choice**: Users opt-in to publishing
- **Selective Sharing**: Users control what to publish
- **Storage Isolation**: Published items link to user storage but don't expose other files

## Future Enhancements

### Planned Features
- [ ] User profiles with published items
- [ ] Comments on published items
- [ ] Advanced search and filters
- [ ] Collections/playlists
- [ ] Remix functionality (copy settings to generator)
- [ ] Report/flag inappropriate content
- [ ] Moderation dashboard
- [ ] Featured content
- [ ] Trending algorithm
- [ ] Follower system

### Technical Improvements
- [ ] Thumbnail generation for videos
- [ ] Image optimization
- [ ] CDN integration
- [ ] Full-text search on prompts
- [ ] Recommendation system
- [ ] Analytics dashboard
- [ ] Rate limiting on likes

## Database Migration

To set up the gallery system, run the migration:

```bash
# In Supabase SQL Editor
-- Run the contents of src/backend/migrations/002_published_media.sql
```

This creates:
- `published_media` table with indexes
- `published_media_likes` table
- RLS policies
- Helper functions (increment_view_count, toggle_like)

## Testing Checklist

- [ ] Publish media from My Media tool
- [ ] View published items in gallery
- [ ] Filter by media type
- [ ] Sort by different criteria
- [ ] Infinite scroll pagination
- [ ] Click item to open detail modal
- [ ] Like item (authenticated)
- [ ] Copy prompt
- [ ] Share link
- [ ] Unpublish item
- [ ] Anonymous users only see SFW
- [ ] Authenticated users can toggle NSFW viewing
- [ ] RLS prevents unauthorized unpublishing
- [ ] View count increments
- [ ] Like count updates correctly

## Monitoring & Metrics

Key metrics to track:
- Total published items
- Publish rate per day
- View count distribution
- Like count distribution
- SFW vs NSFW ratio
- Media type distribution
- Average engagement per item
- User publishing activity
