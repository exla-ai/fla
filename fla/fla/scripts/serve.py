#!/usr/bin/env python3
"""FLA Model Serving Script.

Serve a trained model for inference via HTTP API.

Usage:
    # Start server with checkpoint
    fla-serve --checkpoint ./checkpoints/my_model/30000 --port 8000

    # Query the server
    curl -X POST http://localhost:8000/predict -d @observation.json
"""

import dataclasses
import logging

import tyro

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """Server arguments."""

    checkpoint: str
    """Path to model checkpoint."""

    host: str = "0.0.0.0"
    """Server host."""

    port: int = 8000
    """Server port."""

    action_dim: int = 14
    """Action dimension."""

    num_steps: int = 10
    """Denoising steps for inference."""


def main(args: Args | None = None) -> None:
    """Main server entry point."""
    if args is None:
        args = tyro.cli(Args)

    logger.info(f"Loading checkpoint: {args.checkpoint}")

    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        raise ImportError(
            "FastAPI and uvicorn required for serving. "
            "Install with: pip install fastapi uvicorn"
        )

    import jax
    import numpy as np

    from fla.models import Pi05Model

    # Load model
    model = Pi05Model.from_pretrained(
        args.checkpoint,
        freeze_vision_backbone=True,
        action_dim=args.action_dim,
    )

    # Create API
    app = FastAPI(title="FLA Model Server", version="0.1.0")

    class PredictRequest(BaseModel):
        images: dict[str, list]  # Image arrays
        state: list[float]
        prompt: str = "Complete the task"

    class PredictResponse(BaseModel):
        actions: list[list[float]]

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest):
        """Generate actions from observation."""
        from fla.models.base import Observation

        # Convert request to observation
        images = {k: np.array(v)[np.newaxis] for k, v in request.images.items()}
        image_masks = {k: np.array([True]) for k in images}
        state = np.array(request.state)[np.newaxis]

        obs = Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=None,  # Would need tokenizer
            tokenized_prompt_mask=None,
        )

        # Generate actions
        rng = jax.random.PRNGKey(0)
        actions = model.sample_actions(rng, obs, num_steps=args.num_steps)

        return PredictResponse(
            actions=actions[0].tolist()  # Remove batch dim
        )

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
