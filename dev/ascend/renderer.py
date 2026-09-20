"""Render validated Ascend deployment templates."""

import logging
from typing import Any
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined

logger = logging.getLogger(__name__)


class Renderer:
    """Render Jinja templates with strict missing-variable validation."""

    def __init__(self, template_dir: Path) -> None:
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            undefined=StrictUndefined,
            trim_blocks=True,
        )
        logger.info("renderer_initialized")

    def render(self, template_name: str, **context: Any) -> str:
        """Render one template using the supplied context."""

        template = self.env.get_template(template_name)

        rendered_text = template.render(**context)
        return rendered_text

    def render_to_file(
        self,
        output_file: Path,
        template_name: str,
        **context: Any,
    ) -> None:
        """Render one template directly to an output file."""

        rendered_text = self.render(template_name, **context)
        output_file.write_text(rendered_text)
        logger.info("rendered_template", extra={"output_file": str(output_file)})
