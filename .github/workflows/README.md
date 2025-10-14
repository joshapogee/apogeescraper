# Apogee Retailer Monitor

Automated scraper to collect product reviews and ratings from multiple audio retailer websites.

## 🎯 What It Does

- Reads product URLs from Google Sheets
- Scrapes ratings and review counts from retailer websites
- Writes results back to Google Sheets
- Runs automatically weekly (or on-demand)
- Handles different retailer website structures
- Saves HTML artifacts when parsing fails (for debugging)

## 📊 Supported Retailers

- ✅ Sweetwater
- ✅ Guitar Center (GC)
- ✅ B&H Photo (BH)
- ✅ Vintage King
- ✅ Thomann
- ✅ Apple App Store
- ✅ Yelp

## 🚀 Setup

### Prerequisites

- GitHub account
- Google account
- Google Sheet with product data

### 1. Google Sheets Setup

1. Create a Google Sheet with two tabs:
   - **Input** (columns: `retailer`, `product_name`, `url`)
   - **Retailer Results** (leave empty, auto-created)

2. Create Google Service Account:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create project → Enable "Google Sheets API"
   - Create Service Account → Download JSON key
   - Share your sheet with the service account email (Editor access)

### 2. GitHub Setup

1. Create new repository
2. Add these files:
   - `monitor_retailers.py`
   - `requirements.txt`
   - `.github/workflows/retailer-weekly.yml`

3. Add Repository Secrets (Settings → Secrets):
   - `GOOGLE_SA_JSON`: Entire contents of service account JSON
   - `SHEET_ID`: Google Sheet ID from URL

### 3. Test Run

1. Go to Actions tab
2. Select "Retailer Monitor - Weekly"
3. Click "Run workflow"
4. Choose a specific retailer to test
5. Check results in your Google Sheet

## 🔧 Configuration

### Retailer Settings

Edit `.github/workflows/retailer-weekly.yml` to adjust:

```yaml
- retailer: "Sweetwater"
  use_playwright: "1"      # 1=browser, 0=fast HTTP
  request_delay: "4.0"     # seconds between requests
  scrape_timeout: "90"     # timeout per page
```

### Environment Variables

- `USE_PLAYWRIGHT`: Use browser (1) or HTTP requests (0)
- `REQUEST_DELAY`: Seconds to wait between products
- `SCRAPE_TIMEOUT`: Max seconds per page load
- `DEBUG_ARTIFACTS`: Save HTML when parsing fails (1/0)
- `LOG_LEVEL`: INFO, DEBUG, WARNING

## 🐛 Debugging

### View Logs

1. Actions tab → Select workflow run
2. Click on retailer job
3. View "Run scraper" logs

### Download HTML Artifacts

1. Scroll to bottom of workflow run
2. Download "html-debug-{retailer}.zip"
3. Inspect HTML to fix selectors

### Common Issues

**"No rating data found"**
- Website structure changed
- Bot detection blocking access
- Try enabling `USE_PLAYWRIGHT: "1"`

**Rate limiting / 429 errors**
- Increase `REQUEST_DELAY`
- Reduce `max-parallel` in workflow

**Timeout errors**
- Increase `SCRAPE_TIMEOUT`
- Check if website is slow/down

## 📅 Schedule

Default: Every Monday at 2pm UTC (6am PT / 9am ET)

Edit schedule in workflow file:
```yaml
schedule:
  - cron: "0 14 * * MON"  # Minute Hour Day Month DayOfWeek
```

## 📝 Output Format

Results written to "Retailer Results" sheet:

- `retailer`: Retailer name
- `product_name`: Product name
- `url`: Product URL
- `status`: "ok" or "error"
- `notes`: Status message
- `avg_rating`: Average rating (e.g., 4.5)
- `total_ratings`: Number of ratings
- `review_count`: Number of reviews
- `qa_count`: Q&A count (if available)
- `answers_count`: Answer count (if available)
- `rating_breakdown_json`: Rating distribution
- `last_review_date`: Most recent review date

## 🛡️ Best Practices

1. **Respect robots.txt**: Check each retailer's robots.txt
2. **Rate limiting**: Keep delays reasonable (3-5 seconds)
3. **Monitoring**: Check results regularly for failures
4. **Updates**: Websites change - update selectors as needed

## 🤝 Contributing

When adding new retailers:

1. Add parser function in `monitor_retailers.py`
2. Add to `get_parser()` function
3. Add to workflow matrix
4. Test with manual workflow run

## 📄 License

Use responsibly and in accordance with each retailer's Terms of Service.
