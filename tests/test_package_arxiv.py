import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "package_arxiv.sh"


class ArxivPackageTransactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_builds_and_verifies_under_transaction_paths(self):
        transaction = self.source.index('TRANSACTION_DIR="$(mktemp -d')
        tar_creation = self.source.index('tar czf "$TARBALL"')
        fresh_hash = self.source.index("SMOKE_TEXT_HASH=")
        verify_record = self.source.index('cat > "$VERIFY_FILE"')
        promotion = self.source.index("promote_outputs()")

        self.assertLess(transaction, tar_creation)
        self.assertLess(tar_creation, fresh_hash)
        self.assertLess(fresh_hash, verify_record)
        self.assertLess(verify_record, promotion)

        before_promotion = self.source[:promotion]
        self.assertNotIn('rm -rf "$FINAL_BUILD_DIR"', before_promotion)
        self.assertNotIn('tar czf "$FINAL_TARBALL"', before_promotion)
        self.assertNotIn('cat > "$FINAL_VERIFY_FILE"', before_promotion)

    def test_verification_record_is_promoted_last(self):
        staging_move = self.source.index('mv "$BUILD_DIR" "$FINAL_BUILD_DIR"')
        tar_move = self.source.index('mv "$TARBALL" "$FINAL_TARBALL"')
        verify_move = self.source.index('mv "$VERIFY_FILE" "$FINAL_VERIFY_FILE"')

        self.assertLess(staging_move, tar_move)
        self.assertLess(tar_move, verify_move)

    def test_failed_promotion_restores_previous_outputs(self):
        self.assertIn('mv "$FINAL_BUILD_DIR" "$PREVIOUS_BUILD_DIR"', self.source)
        self.assertIn('mv "$PREVIOUS_BUILD_DIR" "$FINAL_BUILD_DIR"', self.source)
        self.assertIn('mv "$PREVIOUS_TARBALL" "$FINAL_TARBALL"', self.source)
        self.assertIn('mv "$PREVIOUS_VERIFY_FILE" "$FINAL_VERIFY_FILE"', self.source)


if __name__ == "__main__":
    unittest.main()
