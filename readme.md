# Telegram Gold Price Analysis Bot

A Python-based Telegram bot designed to analyze gold prices based on global USD and gold ounce prices. It fetches real-time data from specified Telegram channels, calculates a "fair price" for 18K gold in Iranian Toman, compares it with the market price, and provides buy/hold/sell signals. The bot includes user settings, price history, chart generation, admin tools, and notification features.

## Features

*   **Real-time Analysis:** Fetches USD and gold ounce prices from external Telegram channels (e.g., `@ecogold_ir`, `@tgjucurrency`).
*   **Price Calculation:** Calculates a "fair price" for 18K gold per gram in Toman based on global rates.
*   **Signal Generation:** Provides clear "Buy", "Wait", or "Sell" signals based on the difference between market and fair prices.
*   **Customizable Thresholds:** Users can set their own "Buy" and "Sell" threshold values.
*   **Notification System:** Sends alerts to users when specific conditions (Buy/Sell/Significant Move) are met. (Notifications are checked periodically).
*   **Price Charts:** Generates and sends price comparison charts (Market vs. Fair Price) for the last 24 hours.
*   **Price History:** Allows users to view price history for different timeframes (24h, 7d, 30d) via charts.
*   **Trend Analysis:** Calculates and displays price trends (UPWARD/DOWNWARD/FLAT) and volatility based on recent data (last 6 hours).
*   **Technical Indicators:** Calculates a simplified Relative Strength Index (RSI).
*   **User Settings:** Allows users to manage notification preferences and thresholds.
*   **Admin Panel:** Provides administrators with statistics, user management, database tools, and broadcasting capabilities.
*   **Audit Logging:** Logs user interactions to a private Telegram channel for analysis.
*   **Crawler Service:** A separate service (`crawler_service.py`) fetches and stores price data with technical indicators every 10 minutes for efficient charting and historical analysis.
*   **About Us Section:** Provides information about the bot, price sources, and the creator.

## Prerequisites

*   Python 3.8 or higher
*   A Telegram Bot Token (obtained from [@BotFather](https://t.me/BotFather))
*   Access to the Telegram channels providing USD and Gold prices (e.g., `@ecogold_ir`, `@tgjucurrency`)
*   (Optional) A private Telegram channel for audit logs

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/babakshofficial/Gold-Purchase-Bot.git
    cd Gold-Purchase-Bot
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(The `requirements.txt` file should contain the necessary libraries like `python-telegram-bot`, `requests`, `beautifulsoup4`, `matplotlib`, `numpy`)*

4.  **Set up environment variables:**
    Create a `.env` file in the project root directory and add the following:
    ```env
    BOT_TOKEN=your_telegram_bot_token_here
    PRIVATE_CHANNEL_ID=your_private_channel_id_for_audit_logs
    ADMIN_IDS=comma_separated_list_of_admin_user_ids (e.g., 123456789,987654321)
    ```

## Configuration

*   **Thresholds:** Default thresholds are defined in the code (`DEFAULT_BUY_THRESHOLD`, `DEFAULT_WAIT_THRESHOLD`). Users can adjust these individually via the bot interface.
*   **Notification Types:** Default notification flags are defined (`DEFAULT_NOTIFICATION_FLAGS`).
*   **Data Channels:** The bot fetches data from `GOLD_CHANNEL_USERNAME` and `USD_CHANNEL_USERNAME` defined in the code. Ensure these channels are accessible.
*   **Crawler:** The `crawler_service.py` script needs the same channel details and runs independently to populate the database.

## Usage

### Running the Bot

1.  **Start the Crawler (in a separate terminal/process):**
    ```bash
    python crawler_service.py
    ```
    This service should run continuously to collect data.

2.  **Start the Bot:**
    ```bash
    python main.py
    ```

### Commands

*   `/start`: Initialize the bot and display the main menu.
*   `/gold`: Perform gold price analysis and show the current signal.
*   `/chart`: Generate and send a price comparison chart (last 24h).
*   `/history`: Access the history menu to view charts for different timeframes.
*   `/settings`: Manage notification preferences and thresholds.
*   `/calc`: Start the gold calculation conversation (enter amount in Toman).
*   `/help`: Show the help menu.
*   `/about`: Display information about the bot, sources, and creator.
*   `/admin`: Access the admin panel (admin only).
*   `/stats`: Show bot statistics (admin only).
*   `/health`: Perform a health check (admin only).
*   `/test_audit`: Test audit logging (admin only).
*   `/broadcast`: Start the broadcast message conversation (admin only).

### Admin Panel

The admin panel (accessed via `/admin`) offers:

*   **Statistics:** View user counts, active notifications, price history stats.
*   **User Management:** View user details.
*   **Price Stats:** See latest, average, and range of prices.
*   **Charts:** View admin-specific charts.
*   **Database Management:** View database info, clear old history.
*   **Data Export:** Export user and price history data as CSV.
*   **Broadcast:** Send messages to all users (with targeting options).

## Project Structure

*   `main.py`: Main bot logic, commands, user interactions, settings, charting (using DB data), notifications, admin panel.
*   `crawler_service.py`: Independent script to fetch prices, calculate indicators, and store data in the database.
*   `requirements.txt`: List of Python dependencies.
*   `.env`: Environment variables (not included in version control for security).

## Database

The bot uses `SQLite` to store:

*   **User Settings:** `user_id`, `username`, `first_name`, `notifications`, `notification_flags`, `buy_threshold`, `wait_threshold`.
*   **Price History:** `timestamp`, `tala_price`, `usd_price`, `ounce_price`, `fair_price`, `difference`, `source` ('bot' or 'crawler'), `rsi`, `volatility`, `trend`.

## Troubleshooting

*   **Bot not responding:** Check the console logs for errors. Ensure the `BOT_TOKEN` is correct and the bot is not running elsewhere (causing a `Conflict` error).
*   **Price data not fetching:** Verify the `GOLD_CHANNEL_USERNAME` and `USD_CHANNEL_USERNAME` are correct and the channels are accessible. Check the logs for fetch errors.
*   **Charts not showing:** Ensure `matplotlib` is installed correctly and a compatible font is available (or use English labels in charts as implemented).
*   **Audit logs failing:** Verify `PRIVATE_CHANNEL_ID` and that the bot is an admin of that channel.
*   **Markdown errors:** These are handled with fallbacks, but check the logs for specific problematic characters in user input or dynamic data.
*   **Crawler not running:** Ensure `crawler_service.py` is started and running independently.

## Security

*   Store your `BOT_TOKEN` and other sensitive information in environment variables (`.env` file) and never commit them to the repository.
*   Restrict the `ADMIN_IDS` list to trusted users only.