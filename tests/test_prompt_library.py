from pathlib import Path

from bmscientist.prompt_library import PromptLibrary


def test_prompt_library_parses_markdown_sections_and_substitutes_values(tmp_path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "demo_agent.md").write_text(
        "\n".join(
            [
                "# Demo Agent",
                "",
                "## greet.system",
                "System says hello to $name.",
                "",
                "## greet.user",
                "User prompt for $task.",
            ]
        ),
        encoding="utf-8",
    )

    library = PromptLibrary(base_dir=prompt_dir)

    assert library.render("demo_agent", "greet.system", name="Blake") == "System says hello to Blake."
    assert library.render("demo_agent", "greet.user", task="testing") == "User prompt for testing."


def test_prompt_library_uses_repo_prompt_files():
    library = PromptLibrary(base_dir=Path(__file__).resolve().parents[1] / "src" / "bmscientist" / "prompts" / "agents")

    rendered = library.render(
        "generation_agent",
        "generate_from_meta_review.user",
        research_goal="Goal",
        document_json="{}",
        generation_guidance_json="[]",
        whitespace_gaps_json="[]",
        evidence_payload_json="[]",
        existing_hypotheses_json="[]",
        target_count=3,
    )

    assert "Meta-review guidance" in rendered
    assert "Whitespace gaps" in rendered
