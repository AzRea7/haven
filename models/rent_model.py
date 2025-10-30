from haven.adapters.logging_utils import get_logger

logger = get_logger(__name__)

class RentModel:
    """
    Placeholder interface for unit-level rent prediction.
    Later this will load a trained LightGBM/XGBoost model from disk.

    For now we use a simple heuristic so the rest of the pipeline runs.
    """

    def __init__(self, model_artifact_path: str | None = None):
        self.model = None
        self.model_artifact_path = model_artifact_path

    def predict_unit_rent(
        self,
        bedrooms: float,
        bathrooms: float,
        sqft: float,
        zipcode: str,
        property_type: str
    ) -> float:
        # Dumb baseline heuristic:
        # base + (bed premium) + (bath premium) + (sqft rate)
        base = 500
        bed_component = 400 * bedrooms
        bath_component = 250 * bathrooms
        sqft_component = 1.0 * sqft  # $1/sqft baseline
        type_adj = 1.05 if property_type in ["apartment_unit", "apartment_complex"] else 1.0

        rent = (base + bed_component + bath_component + sqft_component) * type_adj
        logger.info("predict_unit_rent", extra={"context": {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft": sqft,
            "zipcode": zipcode,
            "property_type": property_type,
            "predicted_rent": rent
        }})
        return rent
