import time
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import os
import random

# -------------------------------------------------
# STEP 1: Load Environment Variables
# -------------------------------------------------
print("🔹 Loading environment variables...")
load_dotenv()
print("✅ Environment variables loaded.\n")

# -------------------------------------------------
# STEP 2: Define Helper Function for Stock Symbols
# -------------------------------------------------
def lookup_company_symbol(company: str) -> str:
    symbols = {
        "Infosys": "INFY",
        "Tesla": "TSLA",
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Amazon": "AMZN",
        "Google": "GOOGL"
    }
    print(f"🔍 Looking up symbol for: {company}")
    result = symbols.get(company, "Unknown")
    print(f"✅ Found symbol: {result}\n")
    return result

# -------------------------------------------------
# STEP 3: Create the Stock Data Agent
# -------------------------------------------------
print("🤖 Creating Stock Agent...")
stock_agent = Agent(
    name="Stock Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True
    )],
    instructions=["Fetch stock prices, fundamentals, and analyst recommendations."],
)
print("✅ Stock Agent Ready!\n")

# -------------------------------------------------
# STEP 4: Create the Company Lookup Agent
# -------------------------------------------------
print("🤖 Creating Company Lookup Agent...")
company_lookup_agent = Agent(
    name="Company Lookup Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[lookup_company_symbol],
    instructions=["Find stock symbols for companies based on their names."],
)
print("✅ Company Lookup Agent Ready!\n")

# -------------------------------------------------
# STEP 5: Combine into Finance Team Agent
# -------------------------------------------------
print("👥 Creating Finance Team Agent...")
finance_team = Agent(
    name="Finance Team",
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[stock_agent, company_lookup_agent],
    instructions=["Fetch stock data and company symbols together."],
)
print("✅ Finance Team Ready!\n")

# -------------------------------------------------
# STEP 6: Retry Logic Function
# -------------------------------------------------
def run_with_retry(agent, query, retries=3, delay=10):
    print("🚀 Running query with retry mechanism...\n")
    for attempt in range(1, retries + 1):
        print(f"Attempt {attempt} of {retries}: Sending request → '{query}'")
       
        # Run the agent
        response = agent.run(query)

        # If success
        if response:
            print("✅ Response received successfully!\n")
            return response

        # If not successful → wait and retry
        wait_time = random.uniform(delay, delay + 5)
        print(f"⚠️ No response. Retrying in {wait_time:.2f} seconds...\n")
        time.sleep(wait_time)

    print("❌ Max retries reached. Could not complete the request.\n")
    return None

# -------------------------------------------------
# STEP 7: Run the Finance Query
# -------------------------------------------------
query = "Compare stock data for Apple and Google."
print(f"📊 Running Final Query: {query}\n")

response = run_with_retry(finance_team, query)

# -------------------------------------------------
# STEP 8: Display Results
# -------------------------------------------------
if response:
    print("🧾 FINAL OUTPUT:\n")
    print(response)
else:
    print("🚫 Failed to get a response after all retries.")