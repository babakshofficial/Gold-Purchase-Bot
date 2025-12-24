# 🪙 Gold Price Analysis Telegram Bot

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-blue.svg)](https://telegram.org/)

A sophisticated Telegram bot that provides real-time gold price analysis for the Iranian market, helping users make informed decisions about buying or selling gold based on global ounce prices and live USD exchange rates.

## ✨ Features

### 📊 Core Functionality
- **Real-time Price Analysis** - Fetches live data from Telegram channels and calculates fair market prices
- **Smart Decision Engine** - Provides buy/wait/sell recommendations based on price differences
- **Price Calculator** - Calculate how many grams of gold you can buy with your budget
- **Interactive Charts** - Visual comparison of market prices vs. fair prices (24-hour history)

### 🎛️ User Experience
- **Inline Keyboard Interface** - Modern, button-based interaction (no typing commands!)
- **Persian Language Support** - Fully localized for Iranian users
- **Customizable Settings** - Personal thresholds and notification preferences
- **Multi-post Scanning** - Automatically searches through recent posts to find valid data

### 🔔 Smart Notifications
- **Price Alerts** - Automatic notifications when it's the perfect time to buy
- **Background Monitoring** - Checks prices every 30 minutes
- **User Preferences** - Enable/disable notifications per user

### 👑 Admin Features
- **User Statistics** - Track total users, active notifications, and price records
- **Broadcast Messages** - Send announcements to all users
- **Audit Logging** - Complete conversation logs sent to private channel

### 🗄️ Data Management
- **SQLite Database** - Persistent storage for user settings and price history
- **Price History Tracking** - Stores all price data for trend analysis
- **Automatic Data Recovery** - Recursively searches posts when latest doesn't contain data

## 🎯 Decision Logic

The bot uses a simple yet effective rule based on price difference:

```
var = market_price - fair_price

🟢 var < 100,000 tomans    → BUY (Best time to purchase)
🟡 100,000 ≤ var < 500,000 → WAIT (Monitor the market)
🔴 var ≥ 500,000 tomans    → SELL (Consider selling)
```

Fair price is calculated as: `(USD_rate × Ounce_price) / 41.5`

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- A Telegram Bot Token (from [@BotFather](https://t.me/BotFather))
- Access to a private Telegram channel for audit logs

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/gold-price-bot.git
cd gold-price-bot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
export BOT_TOKEN="your_bot_token_here"
export PRIVATE_CHANNEL_ID="your_channel_id"
export ADMIN_IDS="123456789,987654321"  # Comma-separated admin user IDs
```

4. **Run the bot**
```bash
python main.py
```

## 📦 Dependencies

```txt
python-telegram-bot>=20.0
beautifulsoup4>=4.11.0
requests>=2.28.0
matplotlib>=3.5.0
lxml>=4.9.0
```

### Optional: Price Monitoring
For automatic price alerts, install the job-queue extension:
```bash
pip install "python-telegram-bot[job-queue]"
```

## 🎮 Usage

### User Commands
- `/start` - Initialize bot and show main menu
- `/gold` - Get instant market analysis
- `/chart` - View price comparison chart
- `/calc` - Calculate grams you can buy
- `/settings` - Customize your preferences
- `/help` - Show help information

### Admin Commands
- `/stats` - View bot statistics
- `/broadcast` - Send message to all users

### Inline Menu
The bot features a modern inline keyboard interface:
- 📊 **تحلیل بازار** (Market Analysis)
- 💰 **محاسبه گرم** (Calculate Grams)
- 📈 **نمودار قیمت** (Price Chart)
- ⚙️ **تنظیمات** (Settings)
- ℹ️ **راهنما** (Help)

## 🗂️ Project Structure

```
gold-price-bot/
│
├── main.py                 # Main bot application
├── gold_bot.db            # SQLite database (auto-created)
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── LICENSE               # MIT License
```

## 🗃️ Database Schema

### Users Table
```sql
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    username TEXT,
    first_name TEXT,
    notifications INTEGER DEFAULT 1,
    buy_threshold INTEGER DEFAULT 100000,
    wait_threshold INTEGER DEFAULT 500000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### Price History Table
```sql
CREATE TABLE price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tala_price INTEGER,
    usd_price REAL,
    ounce_price REAL,
    fair_price REAL,
    difference REAL
)
```

## 🔧 Configuration

### Data Sources
The bot scrapes data from these Telegram channels:
- **Gold Prices**: [@ecogold_ir](https://t.me/ecogold_ir)
- **USD Rates**: [@tgjucurrency](https://t.me/tgjucurrency)

### Default Thresholds
```python
DEFAULT_BUY_THRESHOLD = 100_000   # tomans
DEFAULT_WAIT_THRESHOLD = 500_000  # tomans
```

Users can customize these values in their settings.

### Monitoring Interval
```python
MONITOR_INTERVAL = 1800  # seconds (30 minutes)
```

## 🛡️ Error Handling

The bot includes robust error handling:
- **Multi-post Scanning**: Checks up to 10 recent posts if latest post lacks data
- **Processing Messages**: Shows "⏳ در حال دریافت اطلاعات..." while fetching data
- **Graceful Degradation**: Continues working even if price monitoring is unavailable
- **Detailed Logging**: All errors logged with full stack traces

## 📊 Features Overview

| Feature | Status | Description |
|---------|--------|-------------|
| Real-time Analysis | ✅ | Live gold price analysis |
| Price Calculator | ✅ | Calculate grams with your budget |
| Price Charts | ✅ | Visual 24-hour trend graphs |
| Custom Thresholds | ✅ | Personalized buy/sell limits |
| Price Alerts | ✅ | Automatic notifications |
| Admin Panel | ✅ | Statistics & broadcasting |
| Audit Logging | ✅ | Complete conversation tracking |
| Multi-language | 🇮🇷 | Persian (Farsi) |

## 🎨 Screenshots

### Main Menu
The bot greets users with an intuitive inline keyboard interface.

### Market Analysis
Provides comprehensive breakdown of:
- Current USD exchange rate
- Global ounce price
- Market price for 18-karat gold
- Calculated fair price
- Price difference
- Buy/Wait/Sell recommendation

### Price Chart
Visual matplotlib charts showing market price trends compared to fair prices.

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- [ ] Add support for other precious metals (silver, platinum)
- [ ] Implement historical price analysis
- [ ] Add more chart types and timeframes
- [ ] Create web dashboard for statistics
- [ ] Add multi-language support
- [ ] Implement machine learning price predictions

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Created by [@b4bak](https://t.me/b4bak)**

## 🙏 Acknowledgments

- Data sources: [@ecogold_ir](https://t.me/ecogold_ir) and [@tgjucurrency](https://t.me/tgjucurrency)
- Built with [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- Charts powered by [matplotlib](https://matplotlib.org/)

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Contact [@b4bak](https://t.me/b4bak) on Telegram

## ⚠️ Disclaimer

This bot provides information for educational purposes only. Price analysis and recommendations should not be considered as financial advice. Always do your own research and consult with financial professionals before making investment decisions.

---

**Star ⭐ this repository if you find it helpful!**