#!/bin/bash
# Setup mock pyserini module for WebShop environment
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

class MockHit:
    """Mock search hit object."""
    def __init__(self, docid, score, raw_json):
        self.docid = docid
        self.score = score
        self._raw = raw_json

    def raw(self):
        """Return raw JSON string (must be callable per WebShop engine)."""
        return self._raw

class LuceneSearcher:
    """Mock LuceneSearcher for WebShop."""

    def __init__(self, index_dir=None, *args, **kwargs):
        self.index_dir = index_dir
        self._num_docs = 1000

    def search(self, query, k=10, *args, **kwargs):
        """Return mock search results as objects with callable raw()."""
        results = []
        for i in range(min(k, 10)):
            raw_json = '{{"id": "B{:03d}", "title": "Product {} for {}", "description": "A great product matching your search", "price": "${:.2f}", "rating": "{}/5.0"}}'.format(
                i, i, query, 19.99 + i * 10, 4.0 + i * 0.1
            )
            hit = MockHit(
                docid=f'B{i:03d}',
                score=1.0 - (i * 0.1),
                raw_json=raw_json
            )
            results.append(hit)
        return results

    def doc(self, docid):
        """Return mock document by ID."""
        raw_json = '{{"id": "{}", "title": "Product {}", "description": "A great product", "price": "$29.99", "rating": "4.5/5.0"}}'.format(docid, docid)
        return MockHit(docid=docid, score=1.0, raw_json=raw_json)

    def num_docs(self):
        return self._num_docs

    def close(self):
        pass
PYEOF

# encode __init__.py
cat > $SITE_PACKAGES/pyserini/encode/__init__.py << 'PYEOF'
"""Mock pyserini encode module."""

class JsonlCollectionIterator:
    """Mock JsonlCollectionIterator for WebShop."""
    def __init__(self, collection_path=None, *args, **kwargs):
        self.collection_path = collection_path
    def __iter__(self):
        return iter([])
    def __len__(self):
        return 0
PYEOF

echo "Mock pyserini setup complete!"
echo "Verifying installation..."
python -c "import pyserini; print(f'pyserini version: {pyserini.__version__}')"
python -c "from pyserini.search.lucene import LuceneSearcher; s = LuceneSearcher('/tmp/test'); hits = s.search('test'); print(f'Hit docid: {hits[0].docid}, raw: {hits[0].raw()[:50]}...')"
echo "All checks passed!"
