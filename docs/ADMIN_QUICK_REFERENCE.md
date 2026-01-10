# Admin Quick Reference

Quick reference card for Oelala administrators.

## Access Admin Panel

1. Log in with admin account
2. Look for **Admin** section in sidebar (bottom)
3. Click **Admin Panel** (👑 icon)

## Common Tasks

### View All Users
- Main panel shows all users
- Use search box for email/user ID
- Filter by tier (free/pro/vip)
- Click user row to expand details

### Adjust User Credits

**Grant Credits:**
1. Expand user row
2. Click "Adjust Credits"
3. Enter positive amount (e.g., `100`)
4. Enter reason: "Bonus for bug report"
5. Click Confirm

**Deduct Credits:**
1. Expand user row
2. Click "Adjust Credits"
3. Enter negative amount (e.g., `-50`)
4. Enter reason: "Refund processed"
5. Click Confirm

### Change User Tier

1. Expand user row
2. Select tier from dropdown:
   - **Free**: Default tier
   - **Pro**: Premium tier (future: discounts)
   - **VIP**: VIP tier (future: exclusive features)
3. Change is immediate

### Grant VIP Status

1. Expand user row
2. Click "Grant VIP" button
3. Crown (👑) icon appears in user row

### Grant Admin Access

1. Expand user row
2. Click "Grant Admin" button
3. Shield (🛡️) icon appears in user row

⚠️ **Warning:** Only grant admin to trusted users!

### View Transaction History

1. Expand user row
2. Scroll to "Recent Transactions" section
3. Shows:
   - Amount (+ for credits added, - for used)
   - Description/reason
   - Date

## Keyboard Shortcuts

- **Ctrl+F** / **Cmd+F**: Focus search box
- **Esc**: Close modals

## Dashboard Stats

Top of admin panel shows:

- 👥 **Total Users**: All registered users
- 💰 **Credits Issued**: Lifetime credits purchased
- 📈 **Credits Used**: Lifetime credits consumed
- 👑 **VIP Users**: Users with VIP status

## User Row Icons

- 🛡️ **Shield**: User is admin
- 👑 **Crown**: User is VIP
- 💰 **Coins**: Current credit balance

## Tier Colors

- **Gray**: Free tier
- **Purple**: Pro tier
- **Gold**: VIP tier

## Important Reminders

✅ **DO:**
- Always provide a reason when adjusting credits
- Check transaction history before making changes
- Grant admin only to trusted individuals
- Monitor system stats regularly
- Keep audit trail for compliance

❌ **DON'T:**
- Grant large credit amounts without verification
- Remove admin status from active admins
- Delete users (not currently supported)
- Make bulk changes without testing first

## Quick SQL Queries

### Find User by Email
```sql
SELECT uc.*, au.email
FROM user_credits uc
JOIN auth.users au ON au.id = uc.user_id
WHERE au.email ILIKE '%search@example.com%';
```

### View Recent Admin Actions
```sql
SELECT 
  created_at,
  user_id,
  amount,
  description
FROM credit_transactions
WHERE type = 'admin'
ORDER BY created_at DESC
LIMIT 10;
```

### List All Admins
```sql
SELECT au.email, uc.tier, uc.is_vip
FROM user_credits uc
JOIN auth.users au ON au.id = uc.user_id
WHERE uc.is_admin = true;
```

## Troubleshooting

### Can't see Admin Panel
- Verify you're logged in
- Check your `is_admin` flag in database
- Hard refresh browser (Ctrl+Shift+R)

### Credits not updating
- Check transaction log for errors
- Verify amount and reason are provided
- Check backend logs

### Users not loading
- Check internet connection
- Verify backend is running
- Check browser console for errors

## Support

For issues:
1. Check backend logs: `journalctl -u oelala-api -f`
2. Check browser console (F12)
3. Review transaction history
4. Contact support with user ID + timestamp

## Related Docs

- [Full Admin Panel Documentation](./ADMIN_PANEL.md)
- [Migration Guide](./ADMIN_MIGRATION_GUIDE.md)
- [Credits System](./CREDITS.md)
