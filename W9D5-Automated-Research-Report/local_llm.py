from crewai.llms.base_llm import BaseLLM


class DeterministicLocalLLM(BaseLLM):
    def __init__(self):
        super().__init__(model="deterministic-local", provider="custom")

    def call(self, messages, tools=None, callbacks=None,
             available_functions=None, from_task=None,
             from_agent=None, response_model=None):

        prompt = str(messages).lower()

        if "write a structured report" in prompt or "complete markdown research report" in prompt:
            return (
                "## Artificial Intelligence in Healthcare\n\n"
                "### Introduction\n"
                "AI supports healthcare through data-driven applications.\n\n"
                "### Applications\n"
                "- Medical image analysis\n"
                "- Clinical decision support\n"
                "- Patient monitoring\n"
                "- Workflow automation\n\n"
                "### Analysis\n"
                "AI can improve diagnostic support and efficiency. "
                "However, reliable data, clinical validation, privacy, "
                "and human oversight are essential.\n\n"
                "### Conclusion\n"
                "AI offers valuable healthcare opportunities when "
                "implemented responsibly and safely."
            )

        if "analyse the research" in prompt or "analyse" in prompt or "analyze" in prompt:
            return (
                "Analysis: AI can improve diagnostic support and "
                "operational efficiency. Risks include biased data, "
                "privacy concerns, and inaccurate predictions. "
                "Clinical validation and human oversight are necessary."
            )

        return (
            "Research findings: AI is used in healthcare for medical "
            "image analysis, clinical decision support, patient "
            "monitoring, and workflow automation. Key concerns include "
            "privacy, bias, data quality, safety, and regulation."
        )
