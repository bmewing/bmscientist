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
        available_skills_json="[]",
        generation_skill_outputs_json="[]",
        seed_candidates_json="[]",
        existing_hypotheses_json="[]",
        avoided_hypotheses_json="[]",
        target_count=3,
    )

    assert "Meta-review guidance" in rendered
    assert "Whitespace gaps" in rendered


def test_prompt_library_generation_prompt_mentions_candidate_origin_policy():
    library = PromptLibrary(base_dir=Path(__file__).resolve().parents[1] / "src" / "bmscientist" / "prompts" / "agents")

    rendered = library.render(
        "generation_agent",
        "generate.user",
        research_goal="Goal",
        document_json="{}",
        evidence_payload_json="[]",
        available_skills_json="[]",
        generation_skill_outputs_json="[]",
        seed_candidates_json="[]",
        existing_hypotheses_json="[]",
        avoided_hypotheses_json="[]",
        target_hypotheses_generated=3,
    )

    assert "candidate_origin_policy" in rendered
    assert "known_candidate_exclusion_terms" in rendered


def test_prompt_library_research_planning_prompt_mentions_novelty_fields():
    library = PromptLibrary(base_dir=Path(__file__).resolve().parents[1] / "src" / "bmscientist" / "prompts" / "agents")

    rendered = library.render(
        "research_planning_agent",
        "create_research_goal.user",
        raw_goal="Goal",
        target_hypotheses_final=3,
        regions="[]",
        strategic_fit_notes="",
        available_skills_json="[]",
    )

    assert "candidate_origin_policy" in rendered
    assert "novelty_check_policy" in rendered


def test_prompt_library_graph_enrichment_prompt_mentions_skill_context():
    library = PromptLibrary(base_dir=Path(__file__).resolve().parents[1] / "src" / "bmscientist" / "prompts" / "agents")

    rendered = library.render(
        "graph_enrichment_agent",
        "propose.user",
        original_query="Goal",
        evidence_json="[]",
        available_skills_json="[]",
        skill_outputs_json="[]",
    )

    assert "Available enrichment skills" in rendered
    assert "Existing skill outputs tied to these materials/chunks" in rendered
