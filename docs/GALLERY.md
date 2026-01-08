# Gallery System

**Updated**: 2026-01-08
**Status**: ✅ Implemented (PR #78)

---

## Overview

Community gallery for sharing AI-generated content with discovery, likes, and filtering.

## Features

### For Creators
- Publish from My Media with title, description, tags
- NSFW tagging
- Track views and likes
- Unpublish at any time

### For Viewers
- Browse responsive grid
- Filter by media type (all/video/image)
- Sort by newest/popular/most viewed
- Anonymous: SFW only
- Copy prompts to learn

---

## Database Schema

### `published_media` Table
- `id`: UUID primary key
- `user_id`: Reference to auth.users
- `storage_path`: Path to media file
- `title`: Max 100 chars
- `description`: Max 500 chars
- `tags`: Array
- `is_nsfw`: Boolean
- `media_type`: video/image/audio
- `metadata`: JSONB (prompts, settings)
- `view_count`, `like_count`: Engagement stats

### `published_media_likes` Table
Many-to-many user↔media likes.

### RLS Policies
- **Public Read**: SFW content
- **Auth Read**: NSFW content
- **Owner Write**: Own content only

---

## API Endpoints

### Publish
```http
POST /api/gallery/publish
Authorization: Bearer <token>

{
  "storage_path": "video/my-video.mp4",
  "title": "My Creation",
  "tags": ["tag1", "tag2"],
  "is_nsfw": false,
  "media_type": "video"
}
```

### List
```http
GET /api/gallery?media_type=video&sort_by=created_at&page=1
```

### Toggle Like
```http
POST /api/gallery/{media_id}/like
Authorization: Bearer <token>
```

### Unpublish
```http
DELETE /api/gallery/{media_id}
Authorization: Bearer <token>
```

---

## Setup

### 1. Database Migration
Run `src/backend/migrations/002_published_media.sql` in Supabase.

### 2. Environment
```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_role_key
```

### 3. Dependencies
```bash
pip install supabase==2.14.0
```

---

## Frontend Components

| Component | Purpose |
|-----------|---------|
| `PublishModal.jsx` | Publish dialog with form |
| `Gallery.jsx` | Main gallery page |
| `MediaDetailModal.jsx` | Full-screen media viewer |
| `MyMediaTool.jsx` | Publish button integration |

---

## Security

- **SFW Default**: Anonymous = SFW only
- **Auth Required**: Likes, NSFW viewing, publishing
- **Owner RLS**: Only owner can unpublish
- **Storage Isolation**: Links don't expose other files

---

## Future

- [ ] User profiles
- [ ] Comments
- [ ] Collections/playlists
- [ ] Report/flag content
- [ ] Featured/trending
