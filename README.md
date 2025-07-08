# Face Clustering and Retrieval System

A Streamlit-based application for detecting, clustering, and searching faces in images using state-of-the-art ArcFace (InsightFace) embeddings.

## Features

- **Face Detection & Embedding**: Uses only InsightFace (ArcFace) for best-in-class accuracy
- **Face Clustering**: DBSCAN-based clustering with automatic parameter optimization
- **Similarity Search**: FAISS-powered ultra-fast similarity search
- **Interactive UI**: User-friendly Streamlit interface with real-time feedback

## Installation

### Required Installation
```bash
pip install -r requirements.txt
pip install insightface  # For ArcFace embeddings (required)
pip install faiss-cpu    # For fast similarity search (recommended)
```

## Usage

1. **Run the application**:
   ```bash
   streamlit run main.py
   ```

2. **Upload images** containing faces (individual portraits or group photos)

3. **Process images** to detect and cluster faces

4. **Search for similar faces** by uploading a query image

## Common Issues and Solutions

### Issue: "0 clusters (N faces)" - No Clustering Results

**Why this happens:**
- Clustering parameters (eps, min_samples) are too strict
- Poor quality face detections

**Solutions:**

1. **Adjust Clustering Parameters**:
   - Use eps=0.4-0.6, min_samples=2-3 for most datasets

2. **Check Image Quality**:
   - Ensure faces are clearly visible
   - Use well-lit images
   - Avoid extreme angles or occlusions

3. **Use Diverse Images**:
   - Include multiple photos of the same person
   - Vary lighting conditions and angles
   - Include different individuals

### Issue: Misclassified Faces in Similarity Search

**Why this happens:**
- Insufficient training data
- Similarity threshold too low

**Solutions:**
1. Use more diverse training images
2. Adjust similarity thresholds in the search results

## Technical Details

### Embedding & Detection Method:
- **ArcFace (InsightFace)**: State-of-the-art face detection and recognition

### Clustering Algorithm:
- **DBSCAN**: Density-based clustering with automatic parameter optimization
- **Distance Metric**: Cosine distance on normalized embeddings
- **Fallback**: Individual face clusters if clustering fails

## Performance Tips

1. **For Large Datasets**: Install FAISS for faster similarity search
2. **For Better Accuracy**: Use high-quality, diverse images
3. **For Faster Processing**: Use GPU-accelerated libraries when available
4. **For Memory Efficiency**: Process images in batches

## Troubleshooting

### "No faces detected"
- Check if images contain clear, unobstructed faces
- Ensure images are in supported formats (JPG, PNG, BMP)

### "Clustering always fails"
- Use more lenient clustering parameters
- Check image quality

### "Poor similarity search results"
- Use more diverse training data
- Adjust similarity thresholds

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License. 