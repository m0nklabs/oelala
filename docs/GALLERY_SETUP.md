# Gallery System Setup Guide

## Prerequisites

- Supabase account and project
- Backend server (Python/FastAPI)
- Frontend (React/Vite)
- Node.js and npm installed
- Python 3.8+ installed

## Step 1: Database Setup

1. Log into your Supabase project dashboard
2. Navigate to the SQL Editor
3. Copy the contents of `src/backend/migrations/002_published_media.sql`
4. Execute the SQL migration

This will create:
- `published_media` table
- `published_media_likes` table
- Indexes for performance
- RLS policies for security
- Helper functions (increment_view_count, toggle_like)

## Step 2: Environment Variables

Add these environment variables to your backend `.env` file:

```bash
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_role_key_here
SUPABASE_ANON_KEY=your_anon_key_here
SUPABASE_JWT_SECRET=your_jwt_secret_here

# Optional: Debug mode
OELALA_DEBUG=0
```

**Important**: 
- Use the **service role key** for SUPABASE_SERVICE_KEY (has full database access)
- The JWT secret is used for token validation
- Never commit these keys to version control

## Step 3: Install Dependencies

### Backend
```bash
cd src/backend
pip install -r requirements.txt
```

This will install the new `supabase` Python package (version 2.14.0) required for the gallery API.

### Frontend
```bash
cd src/frontend
npm install
```

## Step 4: Start the Backend

```bash
cd src/backend
python app.py
```

The backend should start on port 7998 (or your configured port). Verify the gallery endpoints are available:
- http://localhost:7998/docs (FastAPI Swagger docs)
- Look for `/api/gallery/*` endpoints

## Step 5: Start the Frontend

```bash
cd src/frontend
npm run dev
```

The frontend should start on port 5174. You should now see:
- "Community" section in the sidebar
- "Gallery" link with 🖼️ emoji
- Publish button (upload icon) in My Media next to favorites

## Step 6: Test the System

### Test Publishing
1. Generate some media (image or video)
2. Go to "My Media" → "All"
3. Hover over a media item
4. Click the green upload icon (publish button)
5. Fill in title, description, tags
6. Set NSFW flag if appropriate
7. Click "Publish"

### Test Gallery Viewing
1. Click "Gallery" in the sidebar
2. You should see your published item in the grid
3. Test filters (all/video/image)
4. Test sorting (newest/popular/most viewed)
5. Click on an item to open detail modal

### Test Engagement
1. Like the item (heart button in detail modal)
2. Share the item (copy link button)
3. Copy the prompt (copy button)
4. View count should increment on each view

### Test Anonymous Access
1. Open an incognito/private browsing window
2. Navigate to the gallery
3. Verify only SFW content is visible
4. Verify login prompt for NSFW content

## Step 7: Verify RLS Policies

Test the security policies:

### Test 1: Anonymous SFW Access
```bash
# Should succeed - anonymous can view SFW
curl http://localhost:7998/api/gallery?is_nsfw=false
```

### Test 2: Anonymous NSFW Access
```bash
# Should return empty or filtered results
curl http://localhost:7998/api/gallery?is_nsfw=true
```

### Test 3: Authenticated NSFW Access
```bash
# Should succeed with valid JWT token
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  http://localhost:7998/api/gallery?is_nsfw=true
```

### Test 4: Unpublish (Owner Only)
```bash
# Should succeed if user is owner
curl -X DELETE \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  http://localhost:7998/api/gallery/MEDIA_ID
```

## Troubleshooting

### Issue: "Gallery service unavailable"
- **Cause**: Supabase client not initialized
- **Solution**: Check SUPABASE_URL and SUPABASE_SERVICE_KEY in .env

### Issue: "Failed to publish media"
- **Cause**: Missing authentication or invalid data
- **Solution**: Check browser console for error details, verify JWT token

### Issue: Published items not showing in gallery
- **Cause**: RLS policies blocking access
- **Solution**: Verify RLS policies are set up correctly in Supabase

### Issue: Modal styles broken
- **Cause**: App.css not loaded
- **Solution**: Verify App.css is imported in main.jsx or App.jsx

### Issue: "Module not found: supabase"
- **Cause**: supabase-py not installed
- **Solution**: Run `pip install supabase==2.14.0`

## Monitoring

### Check Backend Logs
```bash
# Look for gallery-related log messages
tail -f backend.log | grep "GALLERY"
```

### Check Database Activity
In Supabase dashboard:
1. Go to Database → Tables
2. View `published_media` table
3. Check row count and recent inserts
4. Monitor `published_media_likes` table

### Check API Performance
Use FastAPI's built-in metrics:
- http://localhost:7998/docs
- Test each endpoint
- Monitor response times

## Production Deployment

### Pre-deployment Checklist
- [ ] Database migration run on production Supabase
- [ ] Environment variables configured
- [ ] Frontend built with `npm run build`
- [ ] Backend dependencies installed
- [ ] RLS policies verified
- [ ] CORS configured for production domain
- [ ] SSL/HTTPS enabled
- [ ] Rate limiting configured
- [ ] Monitoring and logging set up

### Performance Optimization
- Enable CDN for media files
- Set up database connection pooling
- Configure caching headers
- Enable gzip compression
- Optimize images and thumbnails

### Security Checklist
- [ ] Service role key secured
- [ ] JWT secrets rotated regularly
- [ ] CORS restricted to production domains
- [ ] Rate limiting on like endpoint
- [ ] Content moderation workflow ready
- [ ] Backup strategy in place

## Next Steps

After successful setup:
1. Monitor user engagement metrics
2. Gather user feedback
3. Plan content moderation features
4. Consider adding comments
5. Implement trending algorithm
6. Add user profiles
7. Build recommendation system

## Support

For issues or questions:
- Check GALLERY_SYSTEM.md for detailed documentation
- Review code comments in gallery_api.py
- Check FastAPI docs at /docs endpoint
- Review Supabase logs in dashboard
