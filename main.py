import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Any
import warnings
import time
warnings.filterwarnings('ignore')

# Only import InsightFace
import insightface
from insightface.app import FaceAnalysis

# Performance monitoring
@st.cache_data
def get_performance_stats():
    return {"processing_times": [], "search_times": [], "memory_usage": []}

def create_download_button(image_array: np.ndarray, filename: str, button_text: str = "📥 Download Original"):
    """Create a download button for an image array"""
    if image_array is None:
        return None
    
    # Convert numpy array to PIL Image
    if len(image_array.shape) == 3:
        pil_image = Image.fromarray(image_array)
    else:
        pil_image = Image.fromarray(image_array, mode='L')
    
    # Convert to bytes
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format='JPEG', quality=95)
    img_buffer.seek(0)
    
    # Create download button
    return st.download_button(
        label=button_text,
        data=img_buffer.getvalue(),
        file_name=f"original_{filename}",
        mime="image/jpeg"
    )

st.set_page_config(
    page_title="Face Clustering & Retrieval System",
    page_icon="👥",
    layout="wide"
)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    st.warning("FAISS not found. Install with: pip install faiss-cpu")

class FaceDatabase:
    """In-memory database with FAISS for fast similarity search"""
    def __init__(self):
        self.faces = []
        self.embeddings = []
        self.metadata = []
        self.original_images = {}  # Store original images by filename
        self.clusters = None
        self.faiss_index = None
        self.embedding_dimension = None
        self.is_faiss_built = False
        self._embeddings_matrix_cache = None
        self._embeddings_matrix_dirty = True
    
    def add_face(self, embedding: np.ndarray, face_image: np.ndarray, source_filename: str, face_id: int, original_image: np.ndarray = None):
        self.faces.append(face_image)
        self.embeddings.append(embedding)
        self.metadata.append({
            'face_id': face_id,
            'source_file': source_filename,
            'face_number': len(self.faces),
            'embedding_id': len(self.embeddings) - 1
        })
        # Store original image if provided
        if original_image is not None:
            self.original_images[source_filename] = original_image
        self.is_faiss_built = False
        self._embeddings_matrix_dirty = True  # Invalidate cache
    
    def build_faiss_index(self):
        if not FAISS_AVAILABLE or len(self.embeddings) == 0:
            return False
        embeddings_matrix = self.get_embeddings_matrix()
        if embeddings_matrix.size == 0:
            return False
        self.embedding_dimension = embeddings_matrix.shape[1]
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
        # Pre-normalize embeddings for better performance
        normalized_embeddings = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        self.faiss_index.add(normalized_embeddings.astype('float32'))
        self.is_faiss_built = True
        return True
    
    def search_similar_faces(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()
        
        if FAISS_AVAILABLE and self.faiss_index is not None and self.is_faiss_built:
            # Pre-normalize query for consistency
            query_normalized = query_embedding / np.linalg.norm(query_embedding)
            query_normalized = query_normalized.reshape(1, -1).astype('float32')
            similarities, indices = self.faiss_index.search(query_normalized, min(k, len(self.embeddings)))
            
            # Log performance
            search_time = time.time() - start_time
            st.session_state.get('performance_stats', {}).setdefault('search_times', []).append(search_time)
            
            return similarities[0], indices[0]
        else:
            embeddings_matrix = self.get_embeddings_matrix()
            if embeddings_matrix.size == 0:
                return np.array([]), np.array([])
            
            # Use pre-computed normalized embeddings if available
            similarities = cosine_similarity([query_embedding], embeddings_matrix)[0]
            top_indices = np.argsort(similarities)[::-1][:k]
            top_similarities = similarities[top_indices]
            
            # Log performance
            search_time = time.time() - start_time
            st.session_state.get('performance_stats', {}).setdefault('search_times', []).append(search_time)
            
            return top_similarities, top_indices
    
    def get_embeddings_matrix(self) -> np.ndarray:
        if not self.embeddings:
            return np.array([])
        
        # Use cached matrix if available and not dirty
        if self._embeddings_matrix_cache is not None and not self._embeddings_matrix_dirty:
            return self._embeddings_matrix_cache
        
        # Build and cache the matrix
        self._embeddings_matrix_cache = np.vstack(self.embeddings)
        self._embeddings_matrix_dirty = False
        return self._embeddings_matrix_cache
    
    def get_face_count(self) -> int:
        return len(self.faces)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_faces': len(self.faces),
            'embedding_dimension': self.embedding_dimension,
            'faiss_available': FAISS_AVAILABLE,
            'faiss_built': self.is_faiss_built,
            'unique_sources': len(set([meta['source_file'] for meta in self.metadata])) if self.metadata else 0,
            'memory_usage_mb': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        total_mb = 0
        # Face images (assuming average 100x100 RGB)
        total_mb += len(self.faces) * 100 * 100 * 3 / (1024 * 1024)
        # Embeddings (assuming 512-dim float32)
        total_mb += len(self.embeddings) * 512 * 4 / (1024 * 1024)
        # Original images (assuming average 1000x1000 RGB)
        total_mb += len(self.original_images) * 1000 * 1000 * 3 / (1024 * 1024)
        return round(total_mb, 2)
    
    def clear(self):
        self.faces = []
        self.embeddings = []
        self.metadata = []
        self.original_images = {}
        self.clusters = None
        self.faiss_index = None
        self.embedding_dimension = None
        self.is_faiss_built = False
        self._embeddings_matrix_cache = None
        self._embeddings_matrix_dirty = True

class FaceProcessor:
    """Handle face detection, embedding, and clustering with ArcFace (InsightFace)"""
    def __init__(self):
        self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        self._processed_images_cache = {}
    
    @st.cache_data
    def process_image_cached(_self, image_hash: str, image_array: np.ndarray, filename: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Cached version of process_image for repeated processing"""
        return _self._process_image_internal(image_array, filename)
    
    def process_image(self, image: np.ndarray, filename: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        # Create a simple hash for caching
        image_hash = str(hash(image.tobytes()))
        return self.process_image_cached(image_hash, image, filename)
    
    def _process_image_internal(self, image: np.ndarray, filename: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        faces_data = []
        # InsightFace expects BGR format
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        faces = self.face_app.get(image_bgr)
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            face_crop = image[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            # Pre-normalize embedding for consistency
            embedding = face.embedding / np.linalg.norm(face.embedding)
            faces_data.append((face_crop, embedding))
        return faces_data
    
    def cluster_faces(self, embeddings: np.ndarray, eps: float = 0.5, min_samples: int = 2) -> np.ndarray:
        if len(embeddings) < 2:
            return np.array([0] * len(embeddings))
        
        # Pre-normalize embeddings once
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity_matrix = cosine_similarity(normalized_embeddings)
        distance_matrix = 1 - similarity_matrix
        distance_matrix = np.clip(distance_matrix, 0, None)
        
        # Optimized clustering parameters
        clustering_params = [
            (eps, min_samples),
            (eps * 1.5, min_samples),
            (eps * 2.0, min_samples),
            (0.3, 1),
            (0.2, 1),
        ]
        
        best_clustering = None
        best_num_clusters = 0
        
        for eps_try, min_samples_try in clustering_params:
            try:
                clustering = DBSCAN(eps=eps_try, min_samples=min_samples_try, metric='precomputed')
                cluster_labels = clustering.fit_predict(distance_matrix)
                unique_clusters = set(cluster_labels)
                num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
                
                if num_clusters > best_num_clusters and num_clusters > 0:
                    best_clustering = cluster_labels
                    best_num_clusters = num_clusters
                
                # Early stopping if we find a good clustering
                if num_clusters >= 1 and num_clusters <= len(embeddings) // 2:
                    break
            except Exception as e:
                continue
        
        if best_clustering is None:
            best_clustering = np.arange(len(embeddings))
        
        unique_clusters = set(best_clustering)
        num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        noise_count = np.sum(best_clustering == -1)
        st.info(f"🔍 Clustering Results: {num_clusters} clusters, {noise_count} noise points, {len(embeddings)} total faces")
        return best_clustering

def initialize_session_state():
    if 'face_db' not in st.session_state:
        st.session_state.face_db = FaceDatabase()
    if 'processor' not in st.session_state:
        st.session_state.processor = FaceProcessor()
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = False
    if 'performance_stats' not in st.session_state:
        st.session_state.performance_stats = {"processing_times": [], "search_times": [], "memory_usage": []}

def main():
    initialize_session_state()
    st.title("👥 Face Clustering and Retrieval System")
    st.markdown("Upload event photos to build a face database, then capture a face to find similar matches!")
    
    # Performance monitoring sidebar
    with st.sidebar:
        st.header("📊 Performance")
        stats = st.session_state.performance_stats
        if stats['search_times']:
            avg_search_time = np.mean(stats['search_times'])
            st.metric("Avg Search Time", f"{avg_search_time:.3f}s")
        if stats['processing_times']:
            avg_process_time = np.mean(stats['processing_times'])
            st.metric("Avg Process Time", f"{avg_process_time:.3f}s")
    
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.subheader("Clustering Parameters")
    eps = st.sidebar.slider(
        "Clustering Sensitivity (eps)",
        0.1, 1.0, 0.5, 0.1,
        help="Lower values = stricter clustering (fewer, tighter clusters). Higher values = looser clustering (more, broader clusters). Try 0.3-0.7 for best results."
    )
    min_samples = st.sidebar.slider(
        "Minimum Samples per Cluster",
        1, 10, 2,
        help="Minimum faces needed to form a cluster. Lower values = more clusters, higher values = fewer clusters. Use 1-3 for small datasets."
    )
    st.sidebar.success("✅ ArcFace (InsightFace) - SOTA")
    st.sidebar.subheader("⚡ Search Engine")
    if FAISS_AVAILABLE:
        st.sidebar.success("✅ FAISS (Ultra Fast)")
    else:
        st.sidebar.warning("⚠️ NumPy cosine similarity")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("📁 Upload Images")
        uploaded_files = st.file_uploader(
            "Choose image files (JPG, PNG, etc.)",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="Upload individual portraits or group photos"
        )
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} files")
            if st.button("🔄 Process Uploaded Images", type="primary"):
                process_uploaded_images(uploaded_files, eps, min_samples)
        display_database_stats()
    with col2:
        st.header("📷 Face Capture & Search")
        if st.session_state.processed_images:
            capture_and_search_interface()
        else:
            st.info("Please upload and process some images first!")
    if st.session_state.processed_images:
        display_processed_faces()

def process_uploaded_images(uploaded_files, eps, min_samples):
    start_time = time.time()
    face_db = st.session_state.face_db
    processor = st.session_state.processor
    face_db.clear()
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_faces = 0
    st.info(f"🔧 Using embedding method: ArcFace (InsightFace)")
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        image = Image.open(uploaded_file)
        image_array = np.array(image.convert('RGB'))
        faces_data = processor.process_image(image_array, uploaded_file.name)
        
        for j, (face_crop, embedding) in enumerate(faces_data):
            face_db.add_face(
                embedding=embedding,
                face_image=face_crop,
                source_filename=uploaded_file.name,
                face_id=total_faces,
                original_image=image_array
            )
            total_faces += 1
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    if total_faces > 0:
        status_text.text("Clustering faces...")
        embeddings_matrix = face_db.get_embeddings_matrix()
        embedding_norms = np.linalg.norm(embeddings_matrix, axis=1)
        st.info(f"📊 Embedding stats: mean norm={embedding_norms.mean():.4f}, std={embedding_norms.std():.4f}")
        cluster_labels = processor.cluster_faces(embeddings_matrix, eps, min_samples)
        face_db.clusters = cluster_labels
        status_text.text("Building FAISS index for fast search...")
        faiss_built = face_db.build_faiss_index()
        st.session_state.processed_images = True
        unique_clusters = set(cluster_labels)
        num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        noise_count = np.sum(cluster_labels == -1)
        if faiss_built:
            status_text.text(f"✅ Processing complete! Found {total_faces} faces in {num_clusters} clusters ({noise_count} noise). FAISS index ready!")
        else:
            status_text.text(f"✅ Processing complete! Found {total_faces} faces in {num_clusters} clusters ({noise_count} noise).")
        if num_clusters == 0:
            st.warning("💡 **Clustering failed!** Try these solutions:")
            st.markdown("1. **Check clustering parameters**: Increase eps or decrease min_samples in sidebar")
            st.markdown("2. **Check image quality**: Ensure faces are clearly visible and well-lit")
            st.markdown("3. **Use more diverse images**: Include multiple photos of the same person")
    else:
        status_text.text("❌ No faces detected in uploaded images.")
    
    # Log processing time
    processing_time = time.time() - start_time
    st.session_state.performance_stats['processing_times'].append(processing_time)
    progress_bar.empty()

def display_database_stats():
    face_db = st.session_state.face_db
    if face_db.faces:
        st.subheader("📊 Database Statistics")
        stats = face_db.get_stats()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Faces", stats['total_faces'])
        with col2:
            st.metric("Source Images", stats['unique_sources'])
        with col3:
            if face_db.clusters is not None:
                unique_clusters = len(set(face_db.clusters))
                st.metric("Face Clusters", unique_clusters)
            else:
                st.metric("Face Clusters", "Not clustered")
        with col4:
            if stats['embedding_dimension']:
                st.metric("Embedding Dim", stats['embedding_dimension'])
            else:
                st.metric("Embedding Dim", "N/A")
        
        # Memory usage
        if stats['memory_usage_mb'] > 0:
            st.info(f"💾 Estimated Memory Usage: {stats['memory_usage_mb']} MB")
        
        if FAISS_AVAILABLE:
            if stats['faiss_built']:
                st.success("🚀 FAISS Index: Ready for ultra-fast search!")
            else:
                st.info("⚡ FAISS Index: Will be built automatically on first search")
        else:
            st.warning("💡 Install FAISS for faster search: pip install faiss-cpu")

def capture_and_search_interface():
    st.subheader("📷 Capture Face")
    
    # Radio button to choose between upload and camera
    capture_method = st.radio(
        "Choose capture method:",
        ["📁 Upload Image", "📷 Use Camera"],
        horizontal=True
    )
    
    query_array = None
    
    if capture_method == "📁 Upload Image":
        uploaded_query = st.file_uploader(
            "Upload a query face image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="query_face"
        )
        if uploaded_query:
            query_image = Image.open(uploaded_query)
            query_array = np.array(query_image.convert('RGB'))
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(query_image, caption="Uploaded Query Face", width=200)
            search_clicked = col2.button("🔍 Find Similar Faces", type="primary")
            if search_clicked:
                find_similar_faces(query_array)
    
    else:  # Use Camera
        st.info("📷 Camera capture will open in a new tab. Please allow camera permissions.")
        camera_photo = st.camera_input(
            "Take a photo of the face you want to search for",
            key="camera_capture"
        )
        if camera_photo:
            query_image = Image.open(camera_photo)
            query_array = np.array(query_image.convert('RGB'))
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(query_image, caption="Captured Query Face", width=200)
            search_clicked = col2.button("🔍 Find Similar Faces", type="primary")
            if search_clicked:
                find_similar_faces(query_array)

def find_similar_faces(query_image: np.ndarray):
    face_db = st.session_state.face_db
    processor = st.session_state.processor
    if not face_db.faces:
        st.error("No faces in database!")
        return
    faces_data = processor.process_image(query_image, "query")
    if not faces_data:
        st.error("No face detected in query image!")
        return
    query_face, query_embedding = faces_data[0]
    if not face_db.is_faiss_built:
        with st.spinner("Building FAISS index for fast search..."):
            face_db.build_faiss_index()
    top_k = min(12, len(face_db.faces))
    similarities, top_indices = face_db.search_similar_faces(query_embedding, top_k)
    if len(similarities) == 0:
        st.error("No similar faces found!")
        return
    st.subheader("🎯 Similar Faces Found")
    if FAISS_AVAILABLE and face_db.is_faiss_built:
        st.info("🚀 Using FAISS for ultra-fast similarity search")
    else:
        st.info("🔍 Using cosine similarity search")

    # Add bulk download button for all similar faces
    unique_source_files = set()
    for face_idx in top_indices:
        metadata = face_db.metadata[face_idx]
        unique_source_files.add(metadata['source_file'])

    # --- Add noise faces that are highly similar to the query ---
    embeddings_matrix = face_db.get_embeddings_matrix()
    noise_indices = [i for i, c in enumerate(face_db.clusters) if c == -1]
    if noise_indices:
        noise_embeddings = embeddings_matrix[noise_indices]
        # Compute cosine similarity between query and all noise faces
        noise_sims = noise_embeddings @ query_embedding / (np.linalg.norm(noise_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8)
        for idx, sim in zip(noise_indices, noise_sims):
            if sim > 0.7 and idx not in top_indices:
                # Add this noise face to the results for display
                top_indices = np.append(top_indices, idx)
                similarities = np.append(similarities, sim)

    # Sort results by similarity (descending)
    sorted_idx = np.argsort(similarities)[::-1]
    top_indices = top_indices[sorted_idx]
    similarities = similarities[sorted_idx]

    if unique_source_files:
        # Create zip file with all original images
        import zipfile
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for source_file in unique_source_files:
                original_img = face_db.original_images.get(source_file)
                if original_img is not None:
                    # Convert numpy array to PIL Image
                    if len(original_img.shape) == 3:
                        pil_image = Image.fromarray(original_img)
                    else:
                        pil_image = Image.fromarray(original_img, mode='L')
                    # Save to bytes
                    img_buffer = io.BytesIO()
                    pil_image.save(img_buffer, format='JPEG', quality=95)
                    img_buffer.seek(0)
                    # Add to zip
                    zip_file.writestr(f"similar_faces_{source_file}.jpg", img_buffer.getvalue())
        zip_buffer.seek(0)
        # Download button
        st.download_button(
            label=f"📦 Download All Similar Images ({len(unique_source_files)} files)",
            data=zip_buffer.getvalue(),
            file_name="similar_faces_images.zip",
            mime="application/zip"
        )
    cols_per_row = 4
    for i in range(0, len(top_indices), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx_in_results = i + j
            if idx_in_results >= len(top_indices):
                break
            face_idx = top_indices[idx_in_results]
            similarity_score = similarities[idx_in_results]
            with col:
                face_img = face_db.faces[face_idx]
                metadata = face_db.metadata[face_idx]
                st.image(face_img, width=120)
                st.write(f"**Score:** {similarity_score:.4f}")
                st.write(f"**File:** {metadata['source_file'][:20]}...")
                if face_db.clusters is not None:
                    cluster_id = face_db.clusters[face_idx]
                    if cluster_id == -1:
                        st.write(f"**Cluster:** (was noise)")
                    else:
                        st.write(f"**Cluster:** {cluster_id}")
                if similarity_score > 0.8:
                    st.success("🎯 Excellent Match")
                elif similarity_score > 0.6:
                    st.info("✅ Good Match")
                elif similarity_score > 0.4:
                    st.warning("⚠️ Weak Match")
                else:
                    st.error("❌ Poor Match")

def display_processed_faces():
    face_db = st.session_state.face_db
    if not face_db.faces:
        return
    st.header("👥 Processed Faces")
    if face_db.clusters is not None:
        unique_clusters = sorted(set(face_db.clusters))
        valid_clusters = [c for c in unique_clusters if c != -1]
        noise_clusters = [c for c in unique_clusters if c == -1]
        if len(valid_clusters) == 0:
            st.warning("⚠️ No valid clusters found! All faces are marked as noise. This might indicate:")
            st.markdown("- **Clustering parameters too strict**: Try adjusting eps and min_samples in sidebar")
            st.markdown("- **Face detection issues**: Some detected regions might not be actual faces")
            st.markdown("- **Need more diverse images**: Include multiple photos of the same person")
            st.subheader("🔍 All Detected Faces (No Clustering)")
            cols = st.columns(5)
            for i, face_img in enumerate(face_db.faces[:20]):
                with cols[i % 5]:
                    metadata = face_db.metadata[i]
                    st.image(face_img, width=100)
                    st.caption(f"{metadata['source_file']}")
            if len(face_db.faces) > 20:
                st.write(f"... and {len(face_db.faces) - 20} more faces")
            return
        for cluster_id in valid_clusters:
            cluster_indices = [i for i, c in enumerate(face_db.clusters) if c == cluster_id]
            st.subheader(f"👥 Cluster {cluster_id} ({len(cluster_indices)} faces)")
            cols = st.columns(min(5, len(cluster_indices)))
            for i, face_idx in enumerate(cluster_indices[:10]):
                with cols[i % 5]:
                    face_img = face_db.faces[face_idx]
                    metadata = face_db.metadata[face_idx]
                    st.image(face_img, width=100)
                    st.caption(f"{metadata['source_file']}")
            if len(cluster_indices) > 10:
                st.write(f"... and {len(cluster_indices) - 10} more faces")
        if noise_clusters:
            noise_indices = [i for i, c in enumerate(face_db.clusters) if c == -1]
            if noise_indices:
                st.subheader(f"🔍 Noise/Single Faces ({len(noise_indices)} faces)")
                st.info("These faces couldn't be grouped with others. They might be:")
                st.markdown("- Unique individuals not appearing elsewhere")
                st.markdown("- Poor quality face detections")
                st.markdown("- Faces with very different lighting/angles")
                cols = st.columns(min(5, len(noise_indices)))
                embeddings_matrix = face_db.get_embeddings_matrix()
                for i, face_idx in enumerate(noise_indices[:10]):
                    with cols[i % 5]:
                        face_img = face_db.faces[face_idx]
                        metadata = face_db.metadata[face_idx]
                        st.image(face_img, width=100)
                        st.caption(f"{metadata['source_file']}")
                        # --- Manual cluster assignment logic ---
                        # Find best cluster for this noise face
                        noise_embedding = embeddings_matrix[face_idx]
                        best_sim = 0
                        best_cluster = None
                        for cluster_id in valid_clusters:
                            cluster_indices = [j for j, c in enumerate(face_db.clusters) if c == cluster_id]
                            cluster_embeddings = embeddings_matrix[cluster_indices]
                            sims = cluster_embeddings @ noise_embedding / (np.linalg.norm(cluster_embeddings, axis=1) * np.linalg.norm(noise_embedding) + 1e-8)
                            max_sim = np.max(sims)
                            if max_sim > best_sim:
                                best_sim = max_sim
                                best_cluster = cluster_id
                        # If similarity is in the doubtful range, ask user
                        if 0.35 < best_sim < 0.7 and best_cluster is not None:
                            st.warning(f"Possible match with Cluster {best_cluster} (similarity: {best_sim:.2f})")
                            # Show cluster faces
                            cluster_indices = [j for j, c in enumerate(face_db.clusters) if c == best_cluster]
                            st.markdown("**Most similar cluster faces:**")
                            cluster_cols = st.columns(min(5, len(cluster_indices)))
                            for k, cidx in enumerate(cluster_indices[:5]):
                                with cluster_cols[k % 5]:
                                    st.image(face_db.faces[cidx], width=60)
                            # Yes/No buttons with session state flag
                            added_flag = f"added_{face_idx}"
                            if not st.session_state.get(added_flag, False):
                                if st.button(f"Yes, add to Cluster {best_cluster}", key=f"add_{face_idx}"):
                                    face_db.clusters[face_idx] = best_cluster
                                    st.session_state.face_db.clusters = face_db.clusters
                                    st.session_state[added_flag] = True
                                    st.success(f"Face added to Cluster {best_cluster}!")
                                    st.rerun()
                                if st.button(f"No, keep as noise", key=f"keep_{face_idx}"):
                                    st.info("Kept as noise.")
                            else:
                                st.success(f"Face added to Cluster {best_cluster}!")
    else:
        st.subheader("All Detected Faces")
        cols = st.columns(5)
        for i, face_img in enumerate(face_db.faces[:20]):
            with cols[i % 5]:
                metadata = face_db.metadata[i]
                st.image(face_img, width=100)
                st.caption(f"{metadata['source_file']}")
        if len(face_db.faces) > 20:
            st.write(f"... and {len(face_db.faces) - 20} more faces")

if __name__ == "__main__":
    main()