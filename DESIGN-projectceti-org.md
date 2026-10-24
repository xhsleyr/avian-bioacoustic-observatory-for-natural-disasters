---
name: Project CETI
url: https://www.projectceti.org/
colors:
  primary: '#4d1bc7'
  primary-accent: '#4749fe'
  primary-button-hover: '#000000'
  nav-link-hover: '#4f1ccb'
  background: '#ffffff'
  text-primary: '#4e5a7c'
  text-inverse: '#ffffff'
  text-dark: '#000000'
  text-muted: '#a2a7b7'
  surface-dark: '#181b22'
  border-light: '#d6d6d6'
  footer-link-hover-dark: '#4749fe'
typography:
  display:
    family: 'GT Super Display'
    size: 90px
    weight: 400
    line-height: 1.2
  heading1:
    family: 'GT Super Display'
    size: 40px
    weight: 400
    line-height: 1.2
  heading2:
    family: 'GT Super Display'
    size: 30px
    weight: 400
    line-height: 1.2
  body:
    family: 'Sohne'
    size: 14px
    weight: 400
    line-height: 1.5
  caption:
    family: 'Sohne'
    size: 12px
    weight: 400
    line-height: 1.5
  small:
    family: 'Sohne'
    size: 10px
    weight: 400
    line-height: 1.5
spacing:
  base: 8px
  scale: [8, 12, 16, 20, 32, 40, 48, 64, 96, 128]
radius:
  sm: 2px
  full: 100px
elevation:
  z-background: -100
  z-base: 1
  z-slider-nav: 2
  z-slider-arrows: 3
  z-dropdown: 900
  z-cart: 1001
  z-overlay: 10000
  z-max: 10000000
layout:
  max-width: 1280px
motion:
  duration-fast: '0.1s'
  duration-base: '0.2s'
  duration-lightbox: '0.3s'
  easing-standard: 'ease'
  easing-spin: 'linear'
components:
  button-primary:
    bg: '{colors.primary-accent}'
    text: '{colors.text-inverse}'
    radius: '0'
    padding: '9px 30px'
  button-secondary:
    bg: 'transparent'
    text: '{colors.text-dark}'
    border: '1px solid {colors.text-dark}'
    radius: '0'
    padding: '20px 0px'
  button-filter:
    bg: '{colors.text-muted}'
    text: '{colors.text-primary}'
    border: '1px solid {colors.text-primary}'
    radius: '{radius.full}'
    padding: '0px 0px'
  input-text:
    bg: '{colors.background}'
    text: '{colors.text-primary}'
    border: '1px solid {colors.border-light}'
    radius: '0'
    padding: '10px 16px'
  nav-link:
    text: '{colors.text-dark}'
    hover-text: '{colors.nav-link-hover}'
    font-size: '12px'
    font-weight: '400'
---

# Design System Inspired by Project CETI

## 1. Visual Theme & Atmosphere
Project CETI presents a sophisticated and futuristic aesthetic that blends scientific rigor with a sense of wonder. The design leverages a vibrant gradient of purple (`#4d1bc7`), blue (`#4749fe`), and cyan (`#befff7` inferred from screenshot) as a foundational visual element, often appearing as soft, flowing backgrounds behind photorealistic and stylized whale imagery. This gradient creates an immersive, deep-ocean atmosphere, contrasting with stark white content sections that emphasize clarity and data. Typography combines the elegant serif `GT Super Display` for headlines with the clean sans-serif `Sohne` for body text, establishing a clear hierarchy and modern feel. The site also incorporates autoplay video in its hero section, adding dynamic movement and engagement.

The visual language is characterized by a blend of organic forms (whales, gradients) and structured elements (clean layouts, precise typography). Interactive elements, such as buttons and navigation links, are designed with subtle transitions, reinforcing a smooth and responsive user experience. The overall impression is one of cutting-edge research, environmental consciousness, and technological innovation, communicated through a thoughtful and visually rich digital experience.

**Key Characteristics**
-   Gradient backgrounds: Flowing purple, blue, and cyan hues.
-   Typography: `GT Super Display` (serif) for headlines, `Sohne` (sans-serif) for body.
-   Color palette: Deep purple (`#4d1bc7`) and vibrant blue (`#4749fe`) accents.
-   Imagery: Photorealistic whales and scientific data visualizations.
-   Layout: Ample whitespace with full-width sections and 5vw side padding.
-   Motion: Autoplay video hero and subtle CSS transitions on interactive elements.
-   Interaction: Buttons with `0px` radius and distinct hover states.

## 2. Color Palette & Roles
The Project CETI color palette is built around a core set of blues and purples, complemented by a versatile neutral scale to maintain readability and visual balance.

-   **Primary**:
    -   `primary`: `#4d1bc7` (Medium Blue) — The brand's signature deep purple, used for key interactive elements, text accents, and as a dominant color in gradients.
    -   `primary-accent`: `#4749fe` (LTTW Neptune) — A vibrant, electric blue used for primary call-to-action buttons and prominent highlights.
    -   `nav-link-hover`: `#4f1ccb` (RGB 79, 28, 203) — The precise color for navigation link hover states, a slightly darker shade of purple than `primary`.
    -   `primary-button-hover`: `#000000` (Black) — The distinct hover state for the `primary-accent` button, creating a strong visual shift.

-   **Interactive**:
    -   `footer-link-hover-dark`: `#4749fe` (LTTW Neptune) — An inferred hover color for links on dark backgrounds, ensuring visibility against `surface-dark` where `primary-button-hover` (`#000000`) would be invisible.

-   **Neutral Scale**:
    -   `background`: `#ffffff` (White) — The dominant background color for content sections, providing a clean canvas.
    -   `text-primary`: `#4e5a7c` (Dark Slate Blue) — The primary color for body text and general content, offering excellent readability on light backgrounds.
    -   `text-inverse`: `#ffffff` (White) — Used for text on dark backgrounds, such as the hero section or the footer.
    -   `text-dark`: `#000000` (Black) — Used for strong emphasis, some navigation links, and as a hover state for certain interactive elements.
    -   `text-muted`: `#a2a7b7` (Light Slate Grey) — A lighter grey for secondary text, metadata, or disabled states, providing subtle contrast.
    -   `surface-dark`: `#181b22` (Neutral Dark) — A very dark, almost black, background used for the footer area.
    -   `border-light`: `#d6d6d6` (Light Grey) — A subtle grey used for borders, such as input fields.

## 3. Typography Rules
Project CETI employs a distinct typographic pairing to differentiate between display and body content, enhancing both visual appeal and readability.

-   **Font Family**:
    -   Primary Headings: `'GT Super Display', serif`
    -   Body Text: `'Sohne', sans-serif`
    -   Code/Monospace: `'monospace'` (inferred fallback)

-   **Hierarchy**:
    -   **Display**: `GT Super Display` `90px` `400` · line-height `1.2` · tracking `none` · Used for the largest, most impactful hero headlines.
    -   **H1**: `GT Super Display` `40px` `400` · line-height `1.2` · tracking `none` · Main section titles.
    -   **H2**: `GT Super Display` `30px` `400` · line-height `1.2` · tracking `none` · Sub-section titles and key statements.
    -   **Body**: `Sohne` `14px` `400` · line-height `1.5` · tracking `none` · Standard paragraph text and general content.
    -   **Caption**: `Sohne` `12px` `400` · line-height `1.5` · tracking `none` · Secondary information, labels, and small descriptive text.
    -   **Small**: `Sohne` `10px` `400` · line-height `1.5` · tracking `none` · Fine print or least prominent text.
    -   **Code/Mono**: `monospace` `14px` `400` · line-height `1.5` · tracking `none` · For technical content or code snippets (inferred).

-   **Principles**:
    -   Pair the elegant serif `GT Super Display` for high-impact headlines with the clean, modern `Sohne` sans-serif for all body text.
    -   Maintain a consistent `line-height` of `1.2` for all `GT Super Display` headings to ensure tight visual grouping.
    -   Utilize a `line-height` of `1.5` for all `Sohne` body and secondary text for optimal readability.
    -   Prioritize `400` (Regular) weight across most text elements, reserving `700` (Bold) for specific emphasis only (none observed in extracted tokens, inferred from general usage).
    -   Ensure sufficient contrast, especially for `text-primary` (`#4e5a7c`) on `background` (`#ffffff`), which passes AA at 6.82:1.

## 4. Component Stylings

### Buttons
Buttons are designed with sharp, `0px` border radii and clear visual feedback on interaction, using `0.2s` transitions for a smooth experience.

#### Primary Button
A vibrant call-to-action button with a solid background, transforming to black on hover.

```css
.button-primary {
  background-color: var(--lttw-neptune, #4749fe);
  color: var(--white, #ffffff);
  font-family: var(--bodyFamily, 'Sohne'), sans-serif;
  font-size: 12px;
  font-weight: 400;
  padding: 9px 30px;
  border: 1px solid var(--white, #ffffff);
  border-radius: 0;
  cursor: pointer;
  transition: background-color var(--duration-base, 0.2s) var(--easing-standard, ease);
}

.button-primary:hover {
  background-color: var(--black, #000000); /* Extracted from .ci-button:hover */
}

.button-primary:active {
  background-color: var(--black, #000000); /* Inferred from screenshot */
  transform: translateY(1px); /* inferred from screenshot */
}

.button-primary:disabled {
  background-color: var(--light-slate-grey, #a2a7b7); /* inferred from screenshot */
  color: var(--text-primary, #4e5a7c); /* inferred from screenshot */
  border-color: var(--light-slate-grey, #a2a7b7); /* inferred from screenshot */
  cursor: not-allowed;
}
```

#### Secondary Button
A text-based button with a border, often used for navigation or less prominent actions.

```css
.button-secondary {
  background-color: transparent;
  color: var(--black, #222222);
  font-family: var(--bodyFamily, 'Sohne'), sans-serif;
  font-size: 12px;
  font-weight: 400;
  padding: 20px 0px;
  border: 1px solid var(--black, #222222);
  border-radius: 0;
  cursor: pointer;
  transition: color var(--duration-base, 0.2s) var(--easing-standard, ease),
              border-color var(--duration-base, 0.2s) var(--easing-standard, ease);
}

.button-secondary:hover {
  color: var(--medium-blue, #4d1bc7); /* Inferred from nav link hover */
  border-color: var(--medium-blue, #4d1bc7); /* Inferred from nav link hover */
}

.button-secondary:active {
  color: var(--medium-blue, #4d1bc7); /* Inferred from screenshot */
  border-color: var(--medium-blue, #4d1bc7); /* Inferred from screenshot */
  transform: translateY(1px); /* inferred from screenshot */
}

.button-secondary:disabled {
  color: var(--light-slate-grey, #a2a7b7); /* inferred from screenshot */
  border-color: var(--light-slate-grey, #a2a7b7); /* inferred from screenshot */
  cursor: not-allowed;
}
```

#### Button Filter
Small, pill-shaped buttons used for categorization or filtering, often found in content sections.

```css
.button-filter {
  background-color: var(--light-slate-grey, #a2a7b7);
  color: var(--dark-slate-blue, #4e5a7c);
  font-family: var(--bodyFamily, 'Sohne'), sans-serif;
  font-size: 14px;
  font-weight: 400;
  padding: 0px 0px; /* Specific padding not extracted, adjust as needed */
  border: 1px solid var(--dark-slate-blue, #4e5a7c);
  border-radius: 100px; /* Full pill shape */
  min-width: 60px; /* inferred from screenshot */
  height: 32px; /* inferred from screenshot */
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: background-color var(--duration-base, 0.2s) var(--easing-standard, ease),
              color var(--duration-base, 0.2s) var(--easing-standard, ease),
              border-color var(--duration-base, 0.2s) var(--easing-standard, ease);
}

.button-filter:hover {
  background-color: var(--medium-blue, #4d1bc7); /* inferred from screenshot */
  color: var(--white, #ffffff); /* inferred from screenshot */
  border-color: var(--medium-blue, #4d1bc7); /* inferred from screenshot */
}

.button-filter:active {
  background-color: var(--medium-blue, #4d1bc7); /* inferred from screenshot */
  color: var(--white, #ffffff); /* inferred from screenshot */
  border-color: var(--medium-blue, #4d1bc7); /* inferred from screenshot */
  transform: translateY(1px); /* inferred from screenshot */
}

.button-filter:disabled {
  background-color: var(--light-grey, #d6d6d6); /* inferred from screenshot */
  color: var(--text-muted, #a2a7b7); /* inferred from screenshot */
  border-color: var(--light-grey, #d6d6d6); /* inferred from screenshot */
  cursor: not-allowed;
}
```

### Cards & Containers
Project CETI primarily uses full-width content sections with large vertical padding rather than traditional shadowed cards. The main content blocks are either white backgrounds or gradient fills.

#### Content Section
Large, full-width containers for primary content, often featuring ample vertical spacing.
```css
.content-section {
  background-color: var(--white, #ffffff);
  padding: 96px var(--_dimensions---site-padding, 5vw); /* inferred from screenshot */
  border-radius: 0;
  box-shadow: none;
  transition: background-color var(--duration-base, 0.2s) var(--easing-standard, ease);
}

.content-section.gradient-bg {
  background: linear-gradient(135deg, #befff7 0%, #4749fe 50%, #4d1bc7 100%); /* inferred from screenshot */
  color: var(--white, #ffffff);
  padding: 128px var(--_dimensions---site-padding, 5vw); /* inferred from screenshot */
}

/* No specific hover state for content sections */
.content-section:hover {
  /* No visual changes observed */
}
```

### Inputs & Forms
Form elements are minimal, with sharp corners and clear states for focus and disability.

#### Text Input
Standard text input field with a light border, becoming disabled with a grey background.

```css
.input-text {
  background-color: var(--white, #ffffff);
  color: var(--dark-slate-blue, #4e5a7c);
  font-family: var(--bodyFamily, 'Sohne'), sans-serif;
  font-size: 14px;
  font-weight: 400;
  padding: 10px 16px; /* inferred from screenshot */
  border: 1px solid var(--light-grey, #d6d6d6);
  border-radius: 0;
  width: 100%;
  box-sizing: border-box;
  transition: border-color var(--duration-base, 0.2s) var(--easing-standard, ease),
              box-shadow var(--duration-base, 0.2s) var(--easing-standard, ease);
}

.input-text:focus {
  border-color: var(--medium-blue, #4d1bc7);
  outline: 2px solid var(--medium-blue, #4d1bc7); /* inferred from screenshot */
  outline-offset: 0px; /* inferred from screenshot */
}

.input-text:disabled {
  background-color: rgb(238, 238, 238); /* Extracted from pseudoStates */
  color: var(--light-slate-grey, #a2a7b7); /* inferred from screenshot */
  cursor: not-allowed;
  border-color: var(--light-grey, #d6d6d6); /* inferred from screenshot */
}
```

#### Form Label
Labels for form fields use the primary text color and body font.

```css
.form-label {
  color: var(--dark-slate-blue, #4e5a7c);
  font-family: var(--bodyFamily, 'Sohne'), sans-serif;
  font-size: 14px;
  font-weight: 400;
  margin-bottom: 8px; /* inferred from screenshot */
  display: block;
}
```

#### Checkbox/Radio
(none observed in source)

### Navigation
The navigation features a clean, minimalist design with clear hover states and a transparent background that adapts to the hero imagery.

#### Top Navigation Bar
A flexible navigation bar that appears transparent over the hero video and becomes opaque on scroll (inferred).

```css
.top-nav-bar {
  background-color: transparent; /* Default over hero */
  color: var(--white, #ffffff); /* Text color over hero */
  padding: 20px var(--_dimensions---site-padding, 5vw); /* inferred from screenshot */
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  z-index: var(--_dimensions---z-index-2, 100);
  transition: background-color var(--duration-base, 0.2s) var(--easing-standard, ease),
              color var(--duration-base, 0.2s) var(--easing-standard, ease);
}

.top-nav-bar.scrolled {
  background-color: var(--white, #ffffff); /* Inferred for scrolled state */
  color: var(--black, #000000); /* Inferred for scrolled state */
  position: fixed; /* Inferred for scrolled state */
  box-shadow: 0 1px 4px rgba(0,0,0,0.08); /* inferred from screenshot */
}
```

#### Navigation Link
Individual links within the navigation bar, changing color on hover.

```css
.ci-nav-link {
  color: var(--black, #222222); /* On light backgrounds */
  font-family: var(--bodyFamily, 'Sohne'), sans-serif;
  font-size: 12px;
  font-weight: 400;
  text-decoration: none;
  padding: 10px 15px; /* inferred from screenshot */
  transition: color var(--duration-base, 0.2s) var(--easing-standard, ease);
}

.top-nav-bar:not(.scrolled) .ci-nav-link {
  color: var(--white, #ffffff); /* Override for hero transparency */
}

.ci-nav-link:hover {
  color: var(--nav-link-hover, #4f1ccb); /* Extracted from pseudoStates */
}

.ci-nav-link.active,
.ci-nav-link[aria-current="page"] {
  color: var(--medium-blue, #4d1bc7); /* inferred from screenshot */
  font-weight: 700; /* inferred from screenshot */
}
```

#### Dropdown Menu
(none explicitly visible in screenshot, but `w-dropdown` is present in z-index values)

### Links
Links are generally understated, relying on color changes for interactive feedback.

#### Standard Link
Inline text links, typically using the primary text color and changing to the brand's primary purple on hover.

```css
.standard-link {
  color: var(--dark-slate-blue, #4e5a7c);
  text-decoration: underline;
  transition: color var(--duration-base, 0.2s) var(--easing-standard, ease);
}

.standard-link:hover {
  color: var(--medium-blue, #4d1bc7); /* inferred from screenshot */
}

.standard-link:visited {
  color: var(--dark-slate-blue, #4e5a7c); /* inferred from screenshot */
}
```

#### Secondary Link
Links used in contexts like the footer, appearing white on a dark background and changing to a vibrant blue on hover for visibility.

```css
.secondary-link {
  color: var(--white, #ffffff);
  text-decoration: none;
  transition: color var(--duration-base, 0.2s) var(--easing-standard, ease);
}

.secondary-link:hover {
  color: var(--footer-link-hover-dark, #4749fe); /* Inferred for visibility on dark backgrounds */
}

.secondary-link:visited {
  color: var(--white, #ffffff); /* inferred from screenshot */
}
```

### Badges
(none observed in source)

## 5. Layout Principles
Project CETI utilizes a spacious and structured layout, emphasizing content clarity and visual hierarchy through consistent spacing and a defined container width.

-   **Spacing System**:
    -   Base unit: `8px`
    -   Scale: `[8, 12, 16, 20, 32, 40, 48, 64, 96, 128]`
    -   Usage Context:
        -   `8px`: Smallest gaps, inline element spacing.
        -   `12px`: Minor vertical spacing between text elements.
        -   `16px`: Standard spacing for form elements, list items.
        -   `20px`: Padding for small components, internal element spacing.
        -   `32px`: Moderate spacing between distinct content blocks.
        -   `40px`: Larger spacing for component groups.
        -   `48px`: Section sub-headings to content.
        -   `64px`: Vertical spacing between major content sections.
        -   `96px`: Large section padding, hero content to edge.
        -   `128px`: Extra large section padding, especially in hero areas.

-   **Grid & Container**:
    -   Max width: `1280px` (from `--_dimensions---max-site-width`)
    -   Container Padding: `5vw` (from `--dimension-site-padding`) applied horizontally to most content.
    -   Columns: `12` columns (inferred from common web design patterns)
    -   Gutter: `24px` (inferred from common web design patterns)
    -   _Note: The site uses a max-width of `1280px` and a `5vw` site padding, which are extracted values. Column counts and gutter widths are common defaults inferred from the visible layout density._

-   **Whitespace Philosophy**:
    Project CETI employs generous whitespace to create a sense of openness and focus. Large vertical and horizontal padding around content blocks ensures that each element has room to breathe, reducing visual clutter and drawing attention to key messages and imagery. This ample spacing contributes to the site's modern and clean aesthetic, allowing the vibrant gradients and whale photography to stand out.

-   **Border Radius Scale**:
    -   `sm`: `2px` — Used for subtle rounding on minor elements (e.g., internal data visualizations).
    -   `full`: `100px` — Applied to create distinct pill-shaped elements, such as filter buttons.
    -   `0px`: Sharp, unrounded corners are the default for most interactive elements like primary buttons and input fields, contributing to a precise and modern feel.

## 6. Depth & Elevation
Project CETI primarily uses layering and large background elements to create depth, rather than extensive use of box shadows. Z-index values are strategically applied to manage content stacking, particularly for navigation, overlays, and background media.

-   **Background Media (z--100)**: `z-index: -100` — Used for background videos or large visual elements that should appear beneath all content.
-   **Base Content (z-1)**: `z-index: 1` — Standard stacking order for primary content blocks and text.
-   **Slider Navigation (z-2)**: `z-index: 2` — Elements related to slider controls, such as pagination dots.
-   **Slider Arrows (z-3)**: `z-index: 3` — Navigation arrows for content sliders.
-   **Dropdown Menus (z-900)**: `z-index: 900` — Standard stacking for dropdown components to appear above most content.
-   **Shopping Cart (z-1001)**: `z-index: 1001` — Specific for e-commerce cart overlays.
-   **Full Overlays (z-10000)**: `z-index: 10000` — Used for full-screen modals or interactive overlays that cover the entire page.
-   **Maximum (z-10000000)**: `z-index: 10000000` — Reserved for critical, top-level elements that must always be visible (e.g., Webflow badge).

**Shadow Philosophy**:
The design system for Project CETI does not heavily rely on traditional box shadows for elevation. Instead, depth is primarily achieved through the layering of content over dynamic gradient backgrounds and autoplay video elements. When shadows are used, they are subtle and minimal (inferred, e.g., for a scrolled navigation bar), focusing on lifting elements just enough to distinguish them without adding heavy visual weight. The emphasis is on a clean, light aesthetic where elements float over immersive backdrops.

## 7. Do's and Don'ts

### Do's
-   **Do** use `GT Super Display` `400` for all headings (H1, H2, H3), maintaining a `line-height` of `1.2`.
-   **Do** apply `Sohne` `14px` `400` with `line-height: 1.5` for all body text, ensuring readability.
-   **Do** use `primary-accent` (`#4749fe`) for the `Button-Primary` background, with `text-inverse` (`#ffffff`) as the text color.
-   **Do** ensure all `Button-Primary` elements transition to `primary-button-hover` (`#000000`) on hover for a distinct interaction.
-   **Do** use `text-primary` (`#4e5a7c`) on `background` (`#ffffff`); this pair achieves a 6.82:1 contrast ratio, passing WCAG AA.
-   **Do** maintain `0px` `border-radius` for `Button-Primary` and `Input-Text` components for a sharp, modern edge.
-   **Do** use `sm` (`2px`) `border-radius` for subtle rounding on internal data visualization elements.
-   **Do** ensure `Nav-Link` elements change `color` to `nav-link-hover` (`#4f1ccb`) on hover for clear feedback.
-   **Do** apply `5vw` horizontal `padding` to all major content sections within the `1280px` `max-width`.
-   **Do** use `full` (`100px`) `border-radius` for `Button-Filter` components to create distinct pill shapes.

### Don'ts
-   **Don't** use `text-primary` (`#4e5a7c`) on `text-muted` (`#a2a7b7`); this pair has a 2.84:1 contrast ratio and fails WCAG AA.
-   **Don't** introduce `border-radius` values other than `0px`, `2px`, or `100px` without explicit design approval.
-   **Don't** use `GT Super Display` for body text; reserve it exclusively for display and heading roles.
-   **Don't** use `padding` or `margin` values outside of the `8px` spacing scale for consistent layout.
-   **Don't** use `background-color: #000000` for text on `surface-dark` (`#181b22`); it will be invisible. Instead, use `footer-link-hover-dark` (`#4749fe`).
-   **Don't** apply custom `outline` styles to `Input-Text` elements; use the default `outline: 2px solid {colors.primary}` on `:focus`.
-   **Don't** use `font-weight: 700` (Bold) for general body text; `400` (Regular) is the standard for `Sohne`.
-   **Don't** place `Nav-Link` text in `text-primary` (`#4e5a7c`) when the navigation bar is transparent over the hero; use `text-inverse` (`#ffffff`).

## 8. Responsive Behavior
_Note: breakpoints below are measured from the source CSS. These values represent the brand's actual media queries._

-   **Breakpoints**:
    -   **Mobile Small** (~479px): Layouts collapse to single-column; typography adjusts for smaller screens.
    -   **Mobile Large** (~767px): Navigation may transition to a hamburger menu; images scale appropriately.
    -   **Tablet** (~991px): Grid layouts may reconfigure from 3-column to 2-column; larger text scales down.
    -   **Desktop** (min 768px): Standard desktop layout; full navigation visible.
    -   **Desktop Large** (min 1280px): Content adheres to `max-width: 1280px`; increased whitespace.

-   **Touch Targets**:
    -   Ensure all interactive elements, especially buttons and links, have a minimum touch target size of `44px` by `44px` (inferred).
    -   Maintain at least `8px` of clear space around touch targets to prevent accidental taps (inferred).

-   **Collapsing Strategy**:
    -   **Navigation**: The top navigation bar collapses into a hamburger menu on screens up to `991px`.
    -   **Cards**: Content sections, while not traditional cards, stack vertically on mobile breakpoints.
    -   **Typography**: Display and heading font sizes scale down progressively across breakpoints to fit smaller viewports.
    -   **Padding**: Horizontal `5vw` padding is maintained, but vertical section padding may reduce on smaller screens (e.g., `96px` to `64px` or `48px` inferred).
    -   **Forms**: Input fields maintain `100%` width, and labels stack above inputs for clarity on mobile.
    -   **Spacing**: Larger spacing units (e.g., `64px`, `96px`) are typically reduced to `32px` or `40px` on mobile for denser content.

## 9. Agent Prompt Guide

-   **Quick Color Reference**:
    -   `primary`: `#4d1bc7`
    -   `primary-accent`: `#4749fe`
    -   `primary-button-hover`: `#000000`
    -   `nav-link-hover`: `#4f1ccb`
    -   `background`: `#ffffff`
    -   `text-primary`: `#4e5a7c`
    -   `text-inverse`: `#ffffff`
    -   `text-dark`: `#000000`
    -   `text-muted`: `#a2a7b7`
    -   `surface-dark`: `#181b22`
    -   `border-light`: `#d6d6d6`
    -   `footer-link-hover-dark`: `#4749fe`

-   **Iteration Guide**:
    1.  Always use `primary-accent` (`#4749fe`) for `Button-Primary` backgrounds, with `text-inverse` (`#ffffff`) text.
    2.  Ensure `Button-Primary` elements transition to `primary-button-hover` (`#000000`) on hover.
    3.  Set `border-radius: 0px` for all `Button-Primary` and `Input-Text` components.
    4.  Apply `GT Super Display` `400` with `line-height: 1.2` for all display and heading text.
    5.  Use `Sohne` `14px` `400` with `line-height: 1.5` for all body text.
    6.  Utilize the `8px` spacing base with `[8, 12, 16, 20, 32, 40, 48, 64, 96, 128]` px scale for all layout and component spacing.
    7.  Maintain `5vw` horizontal padding for all main content sections, adhering to a `1280px` `max-width`.
    8.  Ensure `Input-Text` fields display `outline: 2px solid {colors.primary}` on `:focus`.
    9.  Implement `0.2s ease` transitions for all interactive elements, such as buttons and navigation links.
    10. Use `full` (`100px`) `border-radius` for pill-shaped `Button-Filter` components.
    11. Prioritize `text-primary` (`#4e5a7c`) on `background` (`#ffffff`) for body content (6.82:1 AA contrast).
    12. For links on dark backgrounds, ensure hover states use `footer-link-hover-dark` (`#4749fe`) for visibility.