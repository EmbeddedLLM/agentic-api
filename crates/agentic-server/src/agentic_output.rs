const BLUE: &str = "\u{1b}[38;5;75m";
const GOLD: &str = "\u{1b}[38;5;214m";
const BRIGHT_WHITE: &str = "\u{1b}[97m";
const MAGENTA: &str = "\u{1b}[38;5;207m";
const BOLD: &str = "\u{1b}[1m";
const RESET: &str = "\u{1b}[0m";

const CODEX_EXAMPLE: &str = "agentic run codex --model Qwen/...";
const CLAUDE_EXAMPLE: &str = "agentic run claude --upstream http://127.0.0.1:8000";

#[must_use]
pub fn render_banner(color: bool) -> String {
    render_box(
        &["⚡  Agentic API", "    Local agent gateway"],
        None,
        BLUE,
        color,
        |row| match row {
            "⚡  Agentic API" if color => {
                format!("{GOLD}⚡{RESET}  {BLUE}{BOLD}Agentic{RESET} {GOLD}{BOLD}API{RESET}")
            }
            "⚡  Agentic API" => row.to_owned(),
            _ if color => format!("{BLUE}{row}{RESET}"),
            _ => row.to_owned(),
        },
    )
}

#[must_use]
pub fn render_help(help: &str, color: bool) -> String {
    let mut rendered = String::with_capacity(help.len() + 512);
    rendered.push_str(&render_banner(color));
    rendered.push_str("\n\n");
    rendered.push_str(help.trim());
    if strip_ansi_codes(help).contains("Usage: agentic <COMMAND>") {
        rendered.push_str("\n\n");
        rendered.push_str(&render_examples(color));
    }
    rendered
}

fn strip_ansi_codes(value: &str) -> String {
    let mut plain = String::with_capacity(value.len());
    let mut in_escape = false;
    for character in value.chars() {
        if in_escape {
            if character.is_ascii_alphabetic() {
                in_escape = false;
            }
        } else if character == '\u{1b}' {
            in_escape = true;
        } else {
            plain.push(character);
        }
    }
    plain
}

#[must_use]
pub fn colorize_help(help: &str, color: bool) -> String {
    if !color {
        return help.to_owned();
    }

    help.lines()
        .map(|line| {
            let line = line.replace("Usage:", &format!("{BLUE}{BOLD}Usage:{RESET}"));
            let line = line.replace("<COMMAND>", &format!("{MAGENTA}{BOLD}<COMMAND>{RESET}"));
            if line.trim_end().ends_with(':') {
                format!("{BLUE}{BOLD}{line}{RESET}")
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[must_use]
pub fn render_examples(color: bool) -> String {
    render_box(&[CODEX_EXAMPLE, CLAUDE_EXAMPLE], Some("Examples"), GOLD, color, |row| {
        if color {
            format!("{BRIGHT_WHITE}{row}{RESET}")
        } else {
            row.to_owned()
        }
    })
}

fn render_box<F>(rows: &[&str], title: Option<&str>, border_color: &str, color: bool, style_row: F) -> String
where
    F: Fn(&str) -> String,
{
    let content_width = rows.iter().map(|row| display_width(row) + 2).max().unwrap_or(2);
    let inner_width = title.map_or(content_width, |title| content_width.max(display_width(title) + 3));
    let border = |text: String| {
        if color {
            format!("{border_color}{text}{RESET}")
        } else {
            text
        }
    };
    let top = match title {
        Some(title) => format!(
            "╭─ {title} {}╮",
            "─".repeat(inner_width.saturating_sub(display_width(title) + 3))
        ),
        None => format!("┌{}┐", "─".repeat(inner_width)),
    };
    let mut lines = vec![border(top)];
    for row in rows {
        let padding = inner_width.saturating_sub(display_width(row) + 2);
        let content = style_row(row);
        lines.push(if color {
            format!(
                "{border_color}│{RESET} {content}{}{border_color} │{RESET}",
                " ".repeat(padding)
            )
        } else {
            format!("│ {content}{} │", " ".repeat(padding))
        });
    }
    lines.push(border(format!("╰{}╯", "─".repeat(inner_width))));
    lines.join("\n")
}

#[must_use]
pub fn redact_url(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        return url.to_owned();
    };
    let suffix_start = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, suffix) = rest.split_at(suffix_start);
    let Some((_userinfo, host)) = authority.rsplit_once('@') else {
        return url.to_owned();
    };
    format!("{scheme}://[REDACTED]@{host}{suffix}")
}

#[must_use]
pub fn redact_urls(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(separator) = remaining.find("://") {
        let scheme_start = remaining[..separator]
            .rfind(|character: char| !(character.is_ascii_alphanumeric() || matches!(character, '+' | '-' | '.')))
            .map_or(0, |index| index + 1);
        let authority_start = separator + 3;
        let url_end = remaining[authority_start..]
            .find(|character: char| character.is_whitespace() || matches!(character, '`' | '"' | '\'' | '<' | '>'))
            .map_or(remaining.len(), |index| authority_start + index);
        let candidate = &remaining[scheme_start..url_end];

        output.push_str(&remaining[..scheme_start]);
        output.push_str(&redact_url(candidate));
        remaining = &remaining[url_end..];
    }
    output.push_str(remaining);
    output
}

fn display_width(value: &str) -> usize {
    value
        .chars()
        .map(|character| usize::from(character != '\u{fe0f}'))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::{colorize_help, redact_url, redact_urls, render_banner, render_examples, render_help};

    #[test]
    fn banner_rows_have_equal_display_width() {
        let banner = render_banner(false);
        let widths: Vec<_> = banner.lines().map(display_width).collect();
        assert!(widths.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn examples_box_rows_have_equal_display_width() {
        let examples = render_examples(false);
        let widths: Vec<_> = examples.lines().map(display_width).collect();
        assert!(widths.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn help_has_logo_and_examples_box() {
        let help = render_help("Usage: agentic <COMMAND>\n\nCommands:", false);
        assert!(help.contains("⚡  Agentic API"));
        assert!(help.contains("╭─ Examples"));
    }

    #[test]
    fn styled_root_usage_keeps_examples_box() {
        let help = render_help("Usage: agentic \u{1b}[35m<COMMAND>\u{1b}[0m", false);

        assert!(help.contains("╭─ Examples"));
    }

    #[test]
    fn help_colors_command_placeholder() {
        let help = colorize_help("Usage: agentic <COMMAND>", true);
        assert!(help.contains("\u{1b}[38;5;207m\u{1b}[1m<COMMAND>"));
        assert!(!colorize_help("Usage: agentic <COMMAND>", false).contains('\u{1b}'));
    }

    #[test]
    fn banner_can_be_colored() {
        let banner = render_banner(true);
        assert!(banner.contains("\u{1b}[38;5;75m"));
        assert!(banner.contains("\u{1b}[38;5;214m"));
        assert!(!render_banner(false).contains('\u{1b}'));
    }

    #[test]
    fn examples_box_can_be_colored() {
        let examples = render_examples(true);
        assert!(examples.contains("\u{1b}[38;5;214m"));
        assert!(examples.contains("\u{1b}[97magentic run codex"));
        assert!(!render_examples(false).contains('\u{1b}'));
    }

    #[test]
    fn redact_url_hides_password() {
        assert_eq!(
            redact_url("postgresql://alice:secret@db.example/agentic"),
            "postgresql://[REDACTED]@db.example/agentic"
        );
        assert_eq!(
            redact_url("postgresql://alice:secret@db.example"),
            "postgresql://[REDACTED]@db.example"
        );
        assert_eq!(
            redact_url("postgresql://alice:secret@db.example?sslmode=require"),
            "postgresql://[REDACTED]@db.example?sslmode=require"
        );
        assert_eq!(
            redact_url("https://secret-token@example.com/path"),
            "https://[REDACTED]@example.com/path"
        );
    }

    #[test]
    fn redact_urls_hides_userinfo_inside_diagnostics() {
        assert_eq!(
            redact_urls("error: invalid value `https://secret-token@example.com?bad=1` for '--upstream'"),
            "error: invalid value `https://[REDACTED]@example.com?bad=1` for '--upstream'"
        );
    }

    fn display_width(value: &str) -> usize {
        value.chars().count()
    }
}
