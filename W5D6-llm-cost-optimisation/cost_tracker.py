import csv
from datetime import date
from pathlib import Path

# Estimated example API pricing in USD per 1 million tokens.
# These are configurable illustrative rates, not live provider prices.
INPUT_COST_PER_MILLION = 0.50
OUTPUT_COST_PER_MILLION = 1.50

# Configurable USD to INR conversion rate.
USD_TO_INR = 88.0

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

LOG_FILE = OUTPUT_DIR / "cost_log.csv"


def calculate_cost(input_tokens, output_tokens):
    input_cost = (
        input_tokens / 1_000_000
    ) * INPUT_COST_PER_MILLION

    output_cost = (
        output_tokens / 1_000_000
    ) * OUTPUT_COST_PER_MILLION

    total_usd = input_cost + output_cost
    total_inr = total_usd * USD_TO_INR

    return round(total_usd, 8), round(total_inr, 6)


def log_request(model, input_tokens, output_tokens):
    today = date.today().isoformat()

    total_usd, total_inr = calculate_cost(
        input_tokens,
        output_tokens
    )

    file_exists = LOG_FILE.exists()

    with LOG_FILE.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Date",
                "Model",
                "Input Tokens",
                "Output Tokens",
                "Total Tokens",
                "Estimated Cost USD",
                "Estimated Cost INR"
            ])

        writer.writerow([
            today,
            model,
            input_tokens,
            output_tokens,
            input_tokens + output_tokens,
            total_usd,
            total_inr
        ])

    print("\nRequest logged successfully.")
    print(f"Model: {model}")
    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Estimated cost: ₹{total_inr:.6f}")


def calculate_daily_total():
    if not LOG_FILE.exists():
        print("No cost records found.")
        return

    today = date.today().isoformat()
    daily_total = 0.0
    total_tokens = 0

    with LOG_FILE.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["Date"] == today:
                daily_total += float(
                    row["Estimated Cost INR"]
                )

                total_tokens += int(row["Total Tokens"])

    print("\n========== DAILY COST SUMMARY ==========")
    print(f"Date: {today}")
    print(f"Total tokens: {total_tokens}")
    print(f"Estimated daily cost: ₹{daily_total:.6f}")
    print("========================================")


if __name__ == "__main__":
    # Sample request records for testing
    log_request("Llama 3.2 (3B)", 500, 200)
    log_request("Qwen 2.5 (3B)", 800, 300)

    calculate_daily_total()