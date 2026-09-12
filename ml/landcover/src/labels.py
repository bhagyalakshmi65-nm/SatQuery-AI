"""
BigEarthNet-19 Land-Cover Label Definitions, Band Metadata, and Mappings.

Based on the recommended 19-class benchmark archive defined by:
Sumbul et al., "BigEarthNet-S2: A Large-Scale Benchmark Archive for
Remote Sensing Image Understanding".
"""

BIGEARTHNET_19_CLASSES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(BIGEARTHNET_19_CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(BIGEARTHNET_19_CLASSES)}
NUM_CLASSES = len(BIGEARTHNET_19_CLASSES)

# Sentinel-2 12 Bands in SeCo-ECO order
SENTINEL2_12_BANDS = [
    {"band": "B01", "name": "Coastal aerosol", "wavelength_nm": 443, "native_gsd_m": 60, "native_grid": (20, 20)},
    {"band": "B02", "name": "Blue", "wavelength_nm": 490, "native_gsd_m": 10, "native_grid": (120, 120)},
    {"band": "B03", "name": "Green", "wavelength_nm": 560, "native_gsd_m": 10, "native_grid": (120, 120)},
    {"band": "B04", "name": "Red", "wavelength_nm": 665, "native_gsd_m": 10, "native_grid": (120, 120)},
    {"band": "B05", "name": "Vegetation red edge 1", "wavelength_nm": 705, "native_gsd_m": 20, "native_grid": (60, 60)},
    {"band": "B06", "name": "Vegetation red edge 2", "wavelength_nm": 740, "native_gsd_m": 20, "native_grid": (60, 60)},
    {"band": "B07", "name": "Vegetation red edge 3", "wavelength_nm": 783, "native_gsd_m": 20, "native_grid": (60, 60)},
    {"band": "B08", "name": "Broad NIR", "wavelength_nm": 842, "native_gsd_m": 10, "native_grid": (120, 120)},
    {"band": "B8A", "name": "Narrow NIR", "wavelength_nm": 865, "native_gsd_m": 20, "native_grid": (60, 60)},
    {"band": "B09", "name": "Water vapour", "wavelength_nm": 940, "native_gsd_m": 60, "native_grid": (20, 20)},
    {"band": "B11", "name": "SWIR 1", "wavelength_nm": 1610, "native_gsd_m": 20, "native_grid": (60, 60)},
    {"band": "B12", "name": "SWIR 2", "wavelength_nm": 2190, "native_gsd_m": 20, "native_grid": (60, 60)},
]

BAND_NAMES_12 = [b["band"] for b in SENTINEL2_12_BANDS]
RGB_BAND_INDICES = [3, 2, 1]  # [B04, B03, B02] for true-color visualization

# Mapping from the original 43 Corine Land Cover (CLC) classes to the 19 BigEarthNet benchmark classes
CLC43_TO_BEN19 = {
    "Continuous urban fabric": "Urban fabric",
    "Discontinuous urban fabric": "Urban fabric",
    "Industrial or commercial units": "Industrial or commercial units",
    "Road and rail networks and associated land": None,
    "Port areas": None,
    "Airports": None,
    "Mineral extraction sites": None,
    "Dump sites": None,
    "Construction sites": None,
    "Green urban areas": "Urban fabric",
    "Sport and leisure facilities": "Urban fabric",
    "Non-irrigated arable land": "Arable land",
    "Permanently irrigated land": "Arable land",
    "Rice fields": "Arable land",
    "Vineyards": "Permanent crops",
    "Fruit trees and berry plantations": "Permanent crops",
    "Olive groves": "Permanent crops",
    "Pastures": "Pastures",
    "Annual crops associated with permanent crops": "Permanent crops",
    "Complex cultivation patterns": "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation": "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas": "Agro-forestry areas",
    "Broad-leaved forest": "Broad-leaved forest",
    "Coniferous forest": "Coniferous forest",
    "Mixed forest": "Mixed forest",
    "Natural grasslands": "Natural grassland and sparsely vegetated areas",
    "Moors and heathland": "Moors, heathland and sclerophyllous vegetation",
    "Sclerophyllous vegetation": "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland-shrub": "Transitional woodland, shrub",
    "Beaches, dunes, sands": "Beaches, dunes, sands",
    "Bare rocks": "Natural grassland and sparsely vegetated areas",
    "Sparsely vegetated areas": "Natural grassland and sparsely vegetated areas",
    "Burnt areas": None,
    "Glaciers and perpetual snow": None,
    "Inland marshes": "Inland wetlands",
    "Peat bogs": "Inland wetlands",
    "Salt marshes": "Coastal wetlands",
    "Salines": "Coastal wetlands",
    "Intertidal flats": "Coastal wetlands",
    "Water courses": "Inland waters",
    "Water bodies": "Inland waters",
    "Coastal lagoons": "Marine waters",
    "Estuaries": "Marine waters",
    "Sea and ocean": "Marine waters",
}


def map_clc_labels_to_ben19(clc_labels: list[str]) -> list[str]:
    """Map a list of CLC-43 labels to BigEarthNet-19 labels."""
    mapped = set()
    for label in clc_labels:
        ben19_label = CLC43_TO_BEN19.get(label)
        if ben19_label and ben19_label in CLASS_TO_IDX:
            mapped.add(ben19_label)
    return sorted(list(mapped))


def labels_to_binary_vector(labels: list[str]) -> list[float]:
    """Convert class names to a binary multi-label indicator vector of length 19."""
    vector = [0.0] * NUM_CLASSES
    for label in labels:
        if label in CLASS_TO_IDX:
            vector[CLASS_TO_IDX[label]] = 1.0
    return vector


def binary_vector_to_labels(vector, threshold: float = 0.5) -> list[dict]:
    """Convert a probability vector to a list of detected classes with confidence scores."""
    results = []
    for idx, prob in enumerate(vector):
        if prob >= threshold:
            results.append({
                "class": IDX_TO_CLASS[idx],
                "confidence": round(float(prob), 4),
            })
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results
