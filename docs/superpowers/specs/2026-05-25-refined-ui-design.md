# Refined UI Design Spec

## Context and Goals

WebLLM Gateway needs one coherent visual system across the primary WebUI and the fallback Gateway console. The selected direction is **A. 纸本编辑部**: a refined, paper-like interface with Source Han Serif SC typography, quiet rules, restrained spacing, and a green-gray action color.

## Design Tokens and Foundations

- Typography: all visible UI text uses `Source Han Serif SC`, with `Noto Serif SC`, `Noto Serif CJK SC`, `STSong`, and `SimSun` as fallbacks. Code blocks and logs also inherit the same family to satisfy the "字体全部统一" requirement.
- Surface: paper background `#f6f4ef`, raised surface `#fffefa`, soft surface `#f2eee5`.
- Text: primary ink `#211f1b`, secondary `#6c675e`, quiet text `#8b8377`.
- Accent: green-gray `#315b55`, hover `#244842`, soft accent `#e5ece9`.
- Semantic colors: success `#47745f`, warning `#a06f38`, danger `#98514e`, info blue `#365979`.
- Radius: 3-6px for controls and panels, keeping the interface crisp and mature.
- Shadow: sparse and soft; prefer border hierarchy over decorative shadows.

## Component Rules

- Header and support strip stay compact and fully clickable, using fine borders instead of blue promo styling.
- Hero keeps the existing product copy and Qwen 3.7 emphasis, but becomes paper-white instead of a dark promotional block.
- Panels, provider rows, account cards, model table, media test, modals, drawers, and advanced pages share the same paper, ink, rule, and accent tokens.
- Fallback console mirrors the same token names and typography so users do not see a second unrelated UI when WebUI assets are unavailable.
- Buttons keep Ant Design behavior but receive refined colors and reduced radius through scoped/global overrides.

## Accessibility Requirements

- Preserve current keyboard flows and button semantics.
- Maintain visible focus rings using the accent color.
- Keep text contrast at or above WCAG AA for body, muted copy, and status text.
- Long labels, model IDs, URLs, and error messages must wrap without overlapping.
- Responsive behavior must keep hero, provider/action columns, media test, and fallback console usable on mobile.

## Content Standards

- Keep current Chinese product wording; do not add explanatory marketing sections.
- Preserve `Qwen 3.7 系列已调通` and the light-task positioning.
- Keep external client names already allowed by product scope: 小龙虾、Hermes、OpenAI、Anthropic.

## Anti-Patterns

- Do not use purple/blue gradients, bokeh, oversized cards, or decorative orbs.
- Do not mix sans-serif and serif UI typography.
- Do not introduce new unrelated hues outside the defined tokens.
- Do not expand the support strip height.

## QA Checklist

- Main WebUI and fallback console both use the self-hosted Source Han/Noto Serif SC font resource.
- Browser screenshots at desktop and mobile widths show no overlapping text.
- `webui` build succeeds and generated assets are updated.
- Gateway restarts on latest code and `/health` reports `ok=true`, `sourceFresh=true`, `sourceStale=false`.
