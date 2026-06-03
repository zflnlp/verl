#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test script for WebShop tools.

This script tests the WebShop tool implementations to ensure
they work correctly before running full training.

Usage:
    python examples/webshop_grpo/test_tools.py

Note: Requires WebShop server to be running at http://localhost:3000
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from verl.tools.function_tool import FunctionTool, load_function_tools_from_path


async def test_function_tools():
    """Test FunctionTool-based WebShop tools."""
    print("=" * 50)
    print("Testing FunctionTool-based WebShop tools")
    print("=" * 50)

    # Load tools
    tool_path = str(Path(__file__).parent / "webshop_tools.py")
    print(f"\nLoading tools from: {tool_path}")

    try:
        tools = load_function_tools_from_path(tool_path)
        print(f"Loaded {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.tool_schema.function.description[:50]}...")
    except Exception as e:
        print(f"ERROR loading tools: {e}")
        return False

    # Test task ID
    test_task_id = "test_task_001"

    # Test 1: Search
    print("\n" + "-" * 40)
    print("Test 1: webshop_search")
    print("-" * 40)
    try:
        search_tool = next(t for t in tools if t.name == "webshop_search")
        result = await search_tool.call({
            "query": "red t-shirt",
            "task_id": test_task_id,
        })
        result_data = json.loads(result)
        print(f"Result: {json.dumps(result_data, indent=2)[:200]}...")
        if "error" in result_data:
            print(f"WARNING: Search returned error: {result_data['error']}")
        else:
            print("✓ Search test passed")
    except Exception as e:
        print(f"ERROR: {e}")
        return False

    # Test 2: Click
    print("\n" + "-" * 40)
    print("Test 2: webshop_click")
    print("-" * 40)
    try:
        click_tool = next(t for t in tools if t.name == "webshop_click")
        result = await click_tool.call({
            "element": "1",
            "task_id": test_task_id,
        })
        result_data = json.loads(result)
        print(f"Result: {json.dumps(result_data, indent=2)[:200]}...")
        if "error" in result_data:
            print(f"WARNING: Click returned error: {result_data['error']}")
        else:
            print("✓ Click test passed")
    except Exception as e:
        print(f"ERROR: {e}")
        return False

    # Test 3: Get Status
    print("\n" + "-" * 40)
    print("Test 3: webshop_get_status")
    print("-" * 40)
    try:
        status_tool = next(t for t in tools if t.name == "webshop_get_status")
        result = await status_tool.call({"task_id": test_task_id})
        result_data = json.loads(result)
        print(f"Result: {json.dumps(result_data, indent=2)[:200]}...")
        if "error" in result_data:
            print(f"WARNING: Status returned error: {result_data['error']}")
        else:
            print("✓ Status test passed")
    except Exception as e:
        print(f"ERROR: {e}")
        return False

    # Test 4: Buy
    print("\n" + "-" * 40)
    print("Test 4: webshop_buy")
    print("-" * 40)
    try:
        buy_tool = next(t for t in tools if t.name == "webshop_buy")
        result = await buy_tool.call({"task_id": test_task_id})
        result_data = json.loads(result)
        print(f"Result: {json.dumps(result_data, indent=2)[:200]}...")
        if "error" in result_data:
            print(f"WARNING: Buy returned error: {result_data['error']}")
        else:
            print("✓ Buy test passed")
    except Exception as e:
        print(f"ERROR: {e}")
        return False

    print("\n" + "=" * 50)
    print("All FunctionTool tests completed!")
    print("=" * 50)
    return True


async def test_reward_function():
    """Test reward function."""
    print("\n" + "=" * 50)
    print("Testing Reward Function")
    print("=" * 50)

    try:
        from reward_function import compute_webshop_reward

        # Test with tool rewards
        ground_truth = {
            "task_id": "test_task_001",
            "goal": "red t-shirt size M",
            "category": "clothing",
            "attributes": {"color": "red", "size": "M"},
        }

        # Test 1: With tool rewards
        print("\nTest 1: Reward with tool rewards")
        extra_info = {"tool_rewards": [0.0, 0.0, 0.8]}
        reward = compute_webshop_reward("test solution", ground_truth, extra_info)
        print(f"  Reward: {reward}")
        assert reward == 0.8, f"Expected 0.8, got {reward}"
        print("  ✓ Passed")

        # Test 2: Without tool rewards (fallback)
        print("\nTest 2: Reward without tool rewards (fallback parsing)")
        reward = compute_webshop_reward(
            "I successfully bought the red t-shirt",
            ground_truth,
        )
        print(f"  Reward: {reward}")
        assert reward > 0, f"Expected positive reward, got {reward}"
        print("  ✓ Passed")

        print("\n" + "=" * 50)
        print("Reward function tests completed!")
        print("=" * 50)
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


async def test_data_preprocessing():
    """Test data preprocessing."""
    print("\n" + "=" * 50)
    print("Testing Data Preprocessing")
    print("=" * 50)

    try:
        from data_preprocess import generate_webshop_tasks, preprocess_webshop_dataset

        # Generate tasks
        print("\nGenerating test tasks...")
        tasks = generate_webshop_tasks(num_tasks=10, seed=42)
        print(f"Generated {len(tasks)} tasks")

        # Show sample task
        print("\nSample task:")
        sample = tasks[0]
        print(f"  Task ID: {sample['task_id']}")
        print(f"  Category: {sample['category']}")
        print(f"  Goal: {sample['goal']}")
        print(f"  Attributes: {sample['attributes']}")

        # Preprocess
        print("\nPreprocessing to DataFrame...")
        df = preprocess_webshop_dataset(tasks, split="test")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Show sample record
        print("\nSample record:")
        sample_row = df.iloc[0]
        print(f"  data_source: {sample_row['data_source']}")
        print(f"  agent_name: {sample_row['agent_name']}")
        print(f"  ability: {sample_row['ability']}")
        print(f"  prompt length: {len(sample_row['prompt'])} messages")

        print("\n" + "=" * 50)
        print("Data preprocessing tests completed!")
        print("=" * 50)
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("WebShop Tools Test Suite")
    print("=" * 60)

    results = []

    # Test 1: Function Tools (requires WebShop server)
    print("\n[1/3] Testing Function Tools...")
    print("NOTE: This test requires WebShop server at http://localhost:3000")
    try:
        result = await test_function_tools()
        results.append(("FunctionTools", result))
    except Exception as e:
        print(f"SKIPPED: {e}")
        results.append(("FunctionTools", None))

    # Test 2: Reward Function
    print("\n[2/3] Testing Reward Function...")
    try:
        result = await test_reward_function()
        results.append(("RewardFunction", result))
    except Exception as e:
        print(f"FAILED: {e}")
        results.append(("RewardFunction", False))

    # Test 3: Data Preprocessing
    print("\n[3/3] Testing Data Preprocessing...")
    try:
        result = await test_data_preprocessing()
        results.append(("DataPreprocessing", result))
    except Exception as e:
        print(f"FAILED: {e}")
        results.append(("DataPreprocessing", False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, result in results:
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        print(f"  {name}: {status}")
    print("=" * 60)

    # Check if any critical tests failed
    critical_failures = [r for n, r in results if r is False]
    if critical_failures:
        print("\n⚠ Some tests failed. Please check the errors above.")
        return 1
    else:
        print("\n✓ All tests passed or skipped!")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
