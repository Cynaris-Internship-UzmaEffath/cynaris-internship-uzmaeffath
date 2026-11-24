import ast
import operator
from typing import Type

from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class CodeExecutionInput(BaseModel):
    expression: str = Field(
        description=(
            "A Python arithmetic or data-analysis expression. "
            "Examples: sum([10, 20, 30]), "
            "round(98.4 / 100 * 100, 2), len([1, 2, 3])."
        )
    )


class SafeExpressionEvaluator(ast.NodeVisitor):
    """Evaluate a restricted set of Python expressions without eval or exec."""

    binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    functions = {
        "sum": sum,
        "min": min,
        "max": max,
        "len": len,
        "abs": abs,
        "round": round,
        "sorted": sorted,
    }

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if not isinstance(node.value, (int, float, str, bool)):
            raise ValueError("Unsupported constant type.")
        return node.value

    def visit_List(self, node):
        return [self.visit(item) for item in node.elts]

    def visit_Tuple(self, node):
        return tuple(self.visit(item) for item in node.elts)

    def visit_Dict(self, node):
        return {
            self.visit(key): self.visit(value)
            for key, value in zip(node.keys, node.values)
        }

    def visit_BinOp(self, node):
        operation = self.binary_ops.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported arithmetic operator.")

        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Exponent is too large.")

        result = operation(left, right)

        if isinstance(result, (int, float)) and abs(result) > 10**12:
            raise ValueError("Result exceeds the permitted limit.")

        return result

    def visit_UnaryOp(self, node):
        operation = self.unary_ops.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported unary operator.")
        return operation(self.visit(node.operand))

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only approved built-in functions are allowed.")

        function = self.functions.get(node.func.id)
        if function is None:
            raise ValueError("Function is not permitted.")

        if node.keywords:
            raise ValueError("Keyword arguments are not supported.")

        arguments = [self.visit(arg) for arg in node.args]
        return function(*arguments)

    def generic_visit(self, node):
        raise ValueError(
            f"Unsupported Python expression: {type(node).__name__}"
        )


class CodeExecutionTool(BaseTool):
    name: str = "Safe Python Code Execution"
    description: str = (
        "Evaluate restricted Python arithmetic and data-analysis expressions. "
        "Supports sum, min, max, len, abs, round and sorted. "
        "Does not execute arbitrary Python scripts, imports or system commands."
    )
    args_schema: Type[BaseModel] = CodeExecutionInput

    def _run(self, expression: str) -> str:
        try:
            if len(expression) > 1000:
                return "Error: Expression is too long."

            tree = ast.parse(expression, mode="eval")
            result = SafeExpressionEvaluator().visit(tree)

            return f"Execution successful. Result: {result}"

        except Exception as error:
            return f"Execution failed: {error}"
