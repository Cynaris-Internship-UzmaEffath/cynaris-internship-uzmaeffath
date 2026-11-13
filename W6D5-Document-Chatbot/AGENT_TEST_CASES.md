# W6D5 - Agent Test Cases

## Objective

Verify that the LangChain agent selects and executes the appropriate tools for different user requests.

## Available Tools

1. **calculator:** Performs basic arithmetic calculations using Python AST operations.
2. **web_search:** Provides simulated information search results.

## Test Cases

| Test | User Request                           | Expected Tool | Actual Result                  | Status |
| ---- | -------------------------------------- | ------------- | ------------------------------ | ------ |
| 1    | Calculate 256 multiplied by 37         | calculator    | 9472                           | PASSED |
| 2    | Search for information about LangChain | web_search    | Returned LangChain information | PASSED |
| 3    | Calculate 150 divided by 5             | calculator    | 30                             | PASSED |

## Verification

- Two tools were registered successfully.
- The agent selected the appropriate tool for all three tasks.
- Tool inputs and outputs were displayed.
- Final responses were generated successfully.

## Conclusion

All three agent test cases passed using the local Ollama model llama3.2:3b.

Note: The web_search tool uses simulated results and does not perform live internet searches.
