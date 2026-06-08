#!/bin/bash
# Setup mock pyserini module for WebShop environment
#
# This script creates a mock pyserini module to avoid PyTorch version conflicts
# when installing WebShop on the server.
#
# Usage:
#   bash examples/webshop_grpo/setup_mock_pyserini.sh

set -e

# Get the Python site-packages path
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "Site-packages: $SITE_PACKAGES"

# Remove existing pyserini
echo "Removing existing pyserini..."
pip uninstall pyserini -y 2>/dev/null || true
rm -rf $SITE_PACKAGES/pyserini*

# Create directory structure
echo "Creating mock pyserini directory structure..."
mkdir -p $SITE_PACKAGES/pyserini/search/lucene
mkdir -p $SITE_PACKAGES/pyserini/encode

# Create __init__.py files
echo "Creating mock pyserini module files..."

# Main __init__.py
cat > $SITE_PACKAGES/pyserini/__init__.py << 'PYEOF'
"""Mock pyserini module for WebShop environment."""
__version__ = "0.0.0-mock"
PYEOF

# search __init__.py
cat > $SITE_PACKAGES/pyserini/search/__init__.py << 'PYEOF'
"""Mock pyserini search module."""
PYEOF

# search/lucene __init__.py with LuceneSearcher mock
cat > $SITE_PACKAGES/pyserini/search/lucene/__init__.py << 'PYEOF'
"""Mock pyserini Lucene search module."""

class LuceneSearcher:
    """Mock LuceneSearcher for WebShop."""

    def __init__(self, index_dir=None, *args, **kwargs):
        self.index_dir = index_dir
        self._num_docs = 1000

    def search(self, query, k=10, *args, **kwargs):
        """Return mock search results."""
        results = []
        for i in range(min(k, 5)):
            result = {
                'docid': f'doc_{i}',
                'score': 1.0 - (i * 0.1),
                'content': f'Mock document {i} for query: {query}',
            }
            results.append(result)
        return results

    def doc(self, docid):
        """Return mock document by ID."""
        class MockDoc:
            def __init__(self, docid):
                self.docid = docid
                self.raw = f'{{"id": "{docid}", "title": "Mock Product", "description": "A mock product for testing", "price": "$29.99", "rating": "4.5/5.0"}}'
                self.contents = f'Mock product {docid}'
        return MockDoc(docid)

    def num_docs(self):
        """Return number of documents."""
        return self._num_docs

    def close(self):
        """Close the searcher."""
        pass
PYEOF

# encode __init__.py with JsonlCollectionIterator mock
cat > $SITE_PACKAGES/pyserini/encode/__init__.py << 'PYEOF'
"""Mock pyserini encode module."""

class JsonlCollectionIterator:
    """Mock JsonlCollectionIterator for WebShop."""

    def __init__(self, collection_path=None, *args, **kwargs):
        self.collection_path = collection_path

    def __iter__(self):
        """Return empty iterator."""
        return iter([])

    def __len__(self):
        """Return 0."""
        return 0
PYEOF

echo "Mock pyserini setup complete!"
echo "Verifying installation..."
python -c "import pyserini; print(f'pyserini version: {pyserini.__version__}')"
python -c "from pyserini.search.lucene import LuceneSearcher; print('LuceneSearcher OK')"
python -c "from pyserini.encode import JsonlCollectionIterator; print('JsonlCollectionIterator OK')"
echo "All checks passed!"
