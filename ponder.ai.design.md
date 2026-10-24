---
version: alpha
name: Ponder

description: >
  Ponder is a minimal, cinematic dark-mode product built for creative professionals.
  The design system is anchored by a near-black background, high-contrast white
  typography, pill-shaped CTAs, and a single blue accent — projecting precision,
  calm authority, and premium craftsmanship.

colors:
  background: "#000000"
  surface: "#171717"
  surface-2: "#252525"
  surface-hover: "#2f2f2f"
  on-surface: "#fafafa"
  on-surface-secondary: "#fafafa94"
  on-surface-tertiary: "#fafafa61"
  border: "#ffffff14"
  border-strong: "#ffffff24"
  primary: "#fafafa"
  on-primary: "#000000"
  secondary: "#252525"
  on-secondary: "#fafafa"
  accent: "#4a9eed"
  success: "#53b365"
  error: "#fb2c36"
  violet: "#8b5cf6"
  black: "#020202"
  white: "#f1f1f1"

typography:
  headline-display:
    fontFamily: Lay Grotesk
    fontFallback: DM Sans
    fontSize: 34px
    fontWeight: "500"
    lineHeight: 40px
    letterSpacing: -0.012em
  headline-lg:
    fontFamily: Lay Grotesk
    fontFallback: DM Sans
    fontSize: 26px
    fontWeight: "500"
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Lay Grotesk
    fontFallback: DM Sans
    fontSize: 22px
    fontWeight: "500"
    lineHeight: 28px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Lay Grotesk
    fontFallback: DM Sans
    fontSize: 18px
    fontWeight: "500"
    lineHeight: 24px
  body-lg:
    fontFamily: Lay Grotesk
    fontFallback: DM Sans
    fontSize: 18px
    fontWeight: "400"
    lineHeight: 28px
  body-md:
    fontFamily: Lay Grotesk
    fontFallback: DM Sans
    fontSize: 16px
    fontWeight: "500"
    lineHeight: 24px
    letterSpacing: -0.005em
  body-sm:
    fontFamily: Lay Grotesk
    fontFallback: DM Sans
    fontSize: 14px
    fontWeight: "450"
    lineHeight: 20px
    letterSpacing: -0.011em
  label-lg:
    fontFamily: Lay Grotesk
    fontFallback: DM Sans
    fontSize: 16px
    fontWeight: "500"
    lineHeight: 16px
    letterSpacing: -0.005em
  label-md:
    fontFamily: Lay Grotesk
    fontFallback: DM Sans
    fontSize: 14px
    fontWeight: "500"
    lineHeight: 14px
  label-sm:
    fontFamily: Lay Grotesk
    fontFallback: DM Sans
    fontSize: 12px
    fontWeight: "500"
    lineHeight: 16px
  mono:
    fontFamily: Fragment Mono
    fontFallback: JetBrains Mono
    fontSize: 14px
    fontWeight: "400"
    lineHeight: 20px
  serif-italic:
    fontFamily: PT Serif
    fontSize: 16px
    fontWeight: "400"
    lineHeight: 24px

rounded:
  none: 0px
  sm: 6px
  md: 10px
  lg: 14px
  xl: 20px
  full: 9999px

spacing:
  xs: 4px
  sm: 12px
  md: 22px
  lg: 40px
  xl: 64px
  gutter: 40px
  margin: 16px
  section: 128px

shadows:
  sm: "0 1px 2px 0 rgba(0,0,0,0.30)"
  md: "0 4px 12px 0 rgba(0,0,0,0.40)"
  lg: "0 8px 24px 0 rgba(0,0,0,0.50)"
  badge: "rgba(0,0,0,0.35) 0px 0px 0px 2px"

components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-primary}"
    typography: "{typography.label-lg}"
    rounded: "{rounded.full}"
    padding: "{spacing.md}"
  button-primary-hover:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-primary}"
    opacity: "0.88"
  button-primary-disabled:
    backgroundColor: "{colors.on-surface-secondary}"
    textColor: "{colors.on-surface}"
    opacity: "0.5"
  button-secondary:
    backgroundColor: "{colors.secondary}"
    textColor: "{colors.on-secondary}"
    typography: "{typography.label-lg}"
    rounded: "{rounded.full}"
    padding: "{spacing.md}"
  button-secondary-hover:
    backgroundColor: "{colors.surface-hover}"
    textColor: "{colors.on-secondary}"
  card:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.on-surface}"
    typography: "{typography.body-md}"
    rounded: "{rounded.lg}"
  card-hover:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.on-surface}"
  input-field:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.on-surface}"
    typography: "{typography.body-sm}"
    rounded: "{rounded.sm}"
    padding: "{spacing.sm}"
  nav-link:
    textColor: "{colors.white}"
    typography: "{typography.label-md}"
  badge:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.primary}"
    rounded: "{rounded.full}"
---

## Overview

Ponder is a precision-crafted, dark-mode-first product experience targeting creative professionals and power users. The design voice is **minimal and cinematic** — high contrast, generous negative space, and a tightly controlled type scale combine to project calm authority and premium quality. There are no decorative gradients or loud palettes; every element earns its place on screen.

The design framework is Tailwind CSS with custom design tokens layered on top via CSS custom properties (`--ds-*`). The resulting system is utilitarian at its core but elevated by the careful selection of a proprietary grotesque typeface, a single vivid accent blue, and sharp pill-shaped CTAs that punctuate the otherwise monochromatic canvas.

## Colors

The palette is built on a near-black foundation with two elevated surface tones and a single chromatic accent.

- **Background:** The page canvas is {colors.background} — pure black — establishing a cinematic, immersive baseline.
- **Surface:** Cards and panels use {colors.surface}, a very dark charcoal one step above the background, to create subtle depth without relying on shadows.
- **Surface-2:** Secondary raised surfaces and button fills use {colors.surface-2}, a slightly lighter neutral for interactive affordance.
- **On-Surface:** Primary text and icons are {colors.on-surface} — near-white, with two dimmed variants ({colors.on-surface-secondary} at ~58% opacity and {colors.on-surface-tertiary} at ~38% opacity) for hierarchy.
- **Accent:** {colors.accent} is the single chromatic highlight — a clear sky blue used for interactive focus rings, links, and key UI indicators.
- **Success / Error:** {colors.success} green and {colors.error} red are used strictly for status communication and never as decorative colors.
- **Violet:** {colors.violet} appears in select illustrative or categorical contexts, not as a primary UI token.
- **Borders:** Borders are expressed as low-opacity white — {colors.border} for default dividers and {colors.border-strong} for slightly more visible separators — to keep surfaces feeling integrated rather than boxed.

## Typography

Ponder uses **Lay Grotesk** as its primary display and UI typeface. It is a proprietary grotesque not available on Google Fonts; the nearest substitute is **DM Sans**. All references to `fontFallback: DM Sans` in the frontmatter indicate this substitution for design mockup purposes.

**Fragment Mono** is used for code, terminal output, and any monospaced content. The nearest Google Fonts substitute is **JetBrains Mono**.

**PT Serif** (italic variants only) appears in editorial or pull-quote contexts, adding a humanist counterpoint to the geometric UI typeface. PT Serif is available on Google Fonts directly.

**YDYoonche** is loaded but appears to be a localization-specific or decorative asset; it is not reflected in the primary type scale.

Type scale structure:

- **Headline Display:** {typography.headline-display.fontSize} / {typography.headline-display.lineHeight}, weight {typography.headline-display.fontWeight} — hero headings, section titles.
- **Headline LG / MD / SM:** Descending steps for sub-section hierarchy.
- **Body MD:** {typography.body-md.fontSize} / {typography.body-md.lineHeight} — the workhorse for card content and descriptions.
- **Body SM:** {typography.body-sm.fontSize} / {typography.body-sm.lineHeight}, weight {typography.body-sm.fontWeight} — supporting text, metadata, nav labels.
- **Label LG / MD / SM:** Used exclusively for interactive elements — buttons, tabs, badges — ensuring legibility at tight sizes.
- **Mono:** {typography.mono.fontSize}, Fragment Mono — code blocks, inline code, and data readouts.

Negative letter-spacing (`-0.005em` to `-0.012em`) is applied to larger sizes to optically tighten headlines, a characteristic of the Lay Grotesk design.

## Layout & Spacing

Layout is governed by a maximum container width of `1240px`, centered with a responsive gutter that scales from `16px` to `40px` via a fluid clamp. Vertical section rhythm uses a similarly fluid value scaling from `72px` to `128px`.

- **Base unit:** 4px — all spacing tokens are multiples of this unit.
- **Horizontal padding:** Buttons and interactive elements use {spacing.md} (22px) horizontal padding to ensure comfortable touch targets.
- **Card internal padding:** Approximately {spacing.sm} to {spacing.md} depending on content density.
- **Gutter:** {spacing.gutter} at full width for column separation.
- **Section rhythm:** {spacing.section} vertical padding between major page sections at desktop breakpoints.

The grid philosophy is generous negative space: never crowd elements, and always let the black canvas breathe between content clusters. This whitespace investment is central to the premium, cinematic tone.

## Elevation & Depth

Depth is expressed through **tonal layering**, not shadow stacking. The surface hierarchy is:

1. {colors.background} — page canvas (z=0)
2. {colors.surface} — cards, panels, input backgrounds (z=1)
3. {colors.surface-2} — secondary buttons, popovers, elevated interactive elements (z=2)

Shadows are used sparingly and only at the component level — for example, the badge uses a `2px` ring shadow (`{shadows.badge}`) rather than a diffuse drop shadow. This restraint keeps the interface feeling flat and precise rather than skeuomorphic.

Interactive state changes (hover on cards, buttons) rely on opacity modulation and border-color shifts rather than shadow growth, preserving the flat visual contract.

## Shapes

The shape language is deliberately **binary**: either perfectly pill-shaped (CTAs, badges, tags) or subtly rounded rectangles (cards, inputs). There is no mid-range ambiguity.

- **Full pill ({rounded.full}):** All primary and secondary buttons, nav pills, and tag elements. This is the brand's most distinctive shape signature.
- **Small radius ({rounded.sm} = 6px):** Input fields and inline interactive chips — enough to soften the rectangle without competing with button pills.
- **Medium radius ({rounded.md} = 10px):** General-purpose intermediate containers.
- **Large radius ({rounded.lg} = 14px):** Feature cards, dialog boxes, and elevated panels — matches the `--ds-radius-lg` token exactly.
- **Sharp ({rounded.none}):** Dividers, code blocks, table cells, and any element that should read as structural rather than interactive.

## Components

### Buttons

The primary button uses {colors.primary} fill with {colors.on-primary} text and a {rounded.full} radius — a crisp white pill against the dark canvas that immediately draws the eye. On hover, opacity reduces to 0.88 to acknowledge the interaction without a color shift. The disabled state uses a low-opacity fill to visually communicate unavailability.

The secondary button uses {colors.secondary} (dark charcoal) fill, making it a quieter sibling to the primary. On hover, the background steps to {colors.surface-hover} for a subtle but clear response.

Both buttons share {typography.label-lg} (16px, weight 500) and identical {spacing.md} horizontal padding, ensuring consistent touch-target sizes across the CTA hierarchy.

### Cards

Cards sit on {colors.surface} with a {rounded.lg} radius and a 1px border at {colors.border} opacity. On hover, the border steps up to {colors.border-strong} opacity — a delicate affordance that keeps the surface feeling alive without introducing motion or shadow. Cards inherit {typography.body-md} for their internal text.

### Input Fields

Inputs share the card surface color {colors.surface} and {rounded.sm} radius, visually grouping them with the panel layer. The border at rest is {colors.border-strong}. Focus state triggers a {colors.accent} outline ring per the `--ds-accent` token. Placeholder text is rendered at the {colors.on-surface-tertiary} opacity level to distinguish it clearly from entered content.

### Navigation Links

Nav links use {typography.label-md} (14px, weight 500) with a 0.75 opacity default, stepping to full opacity on hover. This convention keeps the navigation visually recessed relative to the page headline, letting content dominate.

### Badge

The notification badge is a small circular element using {colors.accent} fill and {colors.primary} text, distinguished by the `{shadows.badge}` ring shadow. It uses {typography.label-sm} for its numeral.

### Monospace Elements

Code and data elements use {typography.mono} (Fragment Mono, 14px) set against the {colors.surface} background with no additional decoration, reinforcing the tool's technical character.

## Do's and Don'ts

**Do's**
- Use {colors.background} as the universal page canvas to maintain the cinematic, immersive baseline.
- Use {rounded.full} for all primary and secondary buttons — the pill shape is the brand's most recognizable interactive signature.
- Apply negative letter-spacing to all headline sizes; {typography.headline-display.letterSpacing} at display level is the reference value.
- Reserve {colors.accent} strictly for focus rings, interactive highlights, and key status indicators — never as a fill color for large surfaces.
- Layer depth using tonal surface steps ({colors.surface}, {colors.surface-2}) rather than shadows, keeping the flat cinematic aesthetic intact.
- Use {typography.mono} for all code, terminal, and data-display contexts to reinforce the product's technical identity.

**Don'ts**
- Do not place {colors.success} or {colors.violet} on large background surfaces — these colors are strictly for status communication and categorical data, not decorative fills.
- Do not use border-radius values outside the defined scale; mixing arbitrary radii breaks the binary pill-vs-rectangle shape contract.
- Do not reduce button border-radius below {rounded.full} for primary CTAs — any lesser value undermines the brand's pill-button signature.
- Do not use font weights below 400 or above 600 in the primary Lay Grotesk scale — weights 400, 450, and 500 define the system's restrained typographic voice.
- Do not introduce warm-toned surfaces or backgrounds — the palette is intentionally cool and neutral; warm tones conflict with the cinematic dark aesthetic.
- Do not use PT Serif outside editorial or pull-quote contexts; mixing serif and grotesque in UI components disrupts typographic cohesion.
