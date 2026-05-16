# test_api.py
# This file tests every endpoint and scenario
# of our SHL Assessment Recommender API.
# Run this while your server is running!

import requests
import json

# Our API base URL
BASE_URL = "http://127.0.0.1:8000"

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_test(name):
    print(f"\n{BLUE}{'='*50}{RESET}")
    print(f"{BLUE}TEST: {name}{RESET}")
    print(f"{BLUE}{'='*50}{RESET}")

def print_pass(message):
    print(f"{GREEN}✅ PASS: {message}{RESET}")

def print_fail(message):
    print(f"{RED}❌ FAIL: {message}{RESET}")

def print_response(response):
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        print(json.dumps(data, indent=2)[:500])  # Show first 500 chars
    except:
        print(response.text[:500])


# -------------------------------------------------------
# TEST 1: Health Check
# -------------------------------------------------------
def test_health():
    print_test("Health Check")
    
    response = requests.get(f"{BASE_URL}/health")
    print_response(response)
    
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "healthy":
            print_pass("Health check returned healthy")
        else:
            print_fail("Health check status not healthy")
    else:
        print_fail(f"Health check failed with status {response.status_code}")


# -------------------------------------------------------
# TEST 2: Home endpoint
# -------------------------------------------------------
def test_home():
    print_test("Home Endpoint")
    
    response = requests.get(f"{BASE_URL}/")
    print_response(response)
    
    if response.status_code == 200:
        print_pass("Home endpoint working")
    else:
        print_fail("Home endpoint failed")


# -------------------------------------------------------
# TEST 3: Get all assessments
# -------------------------------------------------------
def test_get_assessments():
    print_test("Get All Assessments")
    
    response = requests.get(f"{BASE_URL}/assessments")
    
    if response.status_code == 200:
        data = response.json()
        total = data.get("total", 0)
        print(f"Total assessments: {total}")
        if total == 20:
            print_pass(f"All 20 assessments loaded correctly")
        else:
            print_fail(f"Expected 20 assessments, got {total}")
    else:
        print_fail("Get assessments failed")


# -------------------------------------------------------
# TEST 4: Basic chat — software engineer
# -------------------------------------------------------
def test_basic_chat():
    print_test("Basic Chat — Software Engineer")
    
    payload = {
        "message": "I need to hire a software engineer",
        "conversation_id": None,
        "conversation_history": []
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    print_response(response)
    
    if response.status_code == 200:
        data = response.json()
        has_response = "response" in data
        has_recommendations = "recommendations" in data
        has_conv_id = "conversation_id" in data
        has_count = "message_count" in data
        
        if all([has_response, has_recommendations, has_conv_id, has_count]):
            print_pass("Response has all required fields")
        else:
            print_fail("Response missing required fields")
    else:
        print_fail(f"Chat failed with status {response.status_code}")


# -------------------------------------------------------
# TEST 5: Multi-turn conversation
# -------------------------------------------------------
def test_multi_turn():
    print_test("Multi-turn Conversation")
    
    # First message
    payload1 = {
        "message": "I need to hire a sales manager",
        "conversation_id": None,
        "conversation_history": []
    }
    
    response1 = requests.post(f"{BASE_URL}/chat", json=payload1)
    data1 = response1.json()
    conv_id = data1.get("conversation_id")
    
    print(f"Turn 1 — Conv ID: {conv_id}")
    print(f"Response: {data1['response'][:100]}...")
    
    # Second message — follow up
    history = [
        {"role": "user", "content": payload1["message"]},
        {"role": "assistant", "content": data1["response"]}
    ]
    
    payload2 = {
        "message": "They need to handle enterprise clients",
        "conversation_id": conv_id,
        "conversation_history": history
    }
    
    response2 = requests.post(f"{BASE_URL}/chat", json=payload2)
    data2 = response2.json()
    
    print(f"\nTurn 2 — Message count: {data2.get('message_count')}")
    print(f"Response: {data2['response'][:150]}...")
    
    if data2.get("message_count") == 2:
        print_pass("Multi-turn conversation working correctly")
    else:
        print_fail("Message count not incrementing correctly")


# -------------------------------------------------------
# TEST 6: Unrelated question — should be refused
# -------------------------------------------------------
def test_unrelated_question():
    print_test("Unrelated Question — Should Be Refused")
    
    payload = {
        "message": "What is the recipe for chocolate cake?",
        "conversation_id": None,
        "conversation_history": []
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    data = response.json()
    
    print(f"Response: {data['response'][:200]}...")
    
    # Check if the AI refused or redirected
    response_text = data["response"].lower()
    refusal_words = ["only", "assessment", "hiring", "cannot", "help with", "shl"]
    
    if any(word in response_text for word in refusal_words):
        print_pass("AI correctly refused unrelated question")
    else:
        print_fail("AI should have refused unrelated question")


# -------------------------------------------------------
# TEST 7: Personality assessment request
# -------------------------------------------------------
def test_personality_assessment():
    print_test("Personality Assessment Request")
    
    payload = {
        "message": "I need a personality test for a customer service role",
        "conversation_id": None,
        "conversation_history": []
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    data = response.json()
    
    print(f"Response: {data['response'][:200]}...")
    print(f"Recommendations: {len(data['recommendations'])} found")
    
    if response.status_code == 200:
        print_pass("Personality assessment request handled")
    else:
        print_fail("Personality assessment request failed")


# -------------------------------------------------------
# TEST 8: Comparison request
# -------------------------------------------------------
def test_comparison():
    print_test("Comparison Request")
    
    # First get some recommendations
    history = [
        {
            "role": "user",
            "content": "I need assessments for a data analyst"
        },
        {
            "role": "assistant",
            "content": "I recommend Verify - Numerical Reasoning and Verify - Inductive Reasoning"
        }
    ]
    
    payload = {
        "message": "Can you compare those two assessments for me?",
        "conversation_id": "test-comparison-123",
        "conversation_history": history
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    data = response.json()
    
    print(f"Response: {data['response'][:200]}...")
    
    if response.status_code == 200:
        print_pass("Comparison request handled successfully")
    else:
        print_fail("Comparison request failed")


# -------------------------------------------------------
# TEST 9: Finance role assessment
# -------------------------------------------------------
def test_finance_role():
    print_test("Finance Role Assessment")
    
    payload = {
        "message": "Looking for numerical and analytical assessments for a finance manager position",
        "conversation_id": None,
        "conversation_history": []
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    data = response.json()
    
    print(f"Recommendations:")
    for rec in data.get("recommendations", []):
        print(f"  - {rec['name']} ({rec['test_type']})")
    
    if response.status_code == 200:
        print_pass("Finance role assessment handled")
    else:
        print_fail("Finance role assessment failed")


# -------------------------------------------------------
# RUN ALL TESTS
# -------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{YELLOW}{'='*50}{RESET}")
    print(f"{YELLOW}SHL API TEST SUITE{RESET}")
    print(f"{YELLOW}{'='*50}{RESET}")
    print("Make sure your server is running before testing!")
    print(f"Testing: {BASE_URL}")
    
    test_health()
    test_home()
    test_get_assessments()
    test_basic_chat()
    test_multi_turn()
    test_unrelated_question()
    test_personality_assessment()
    test_comparison()
    test_finance_role()
    
    print(f"\n{YELLOW}{'='*50}{RESET}")
    print(f"{YELLOW}ALL TESTS COMPLETE!{RESET}")
    print(f"{YELLOW}{'='*50}{RESET}")