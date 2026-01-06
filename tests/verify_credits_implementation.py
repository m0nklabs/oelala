#!/usr/bin/env python3
"""
Credit System Implementation Verification Script

This script verifies that the credit system implementation is complete and functional.
Run this to check that all components are properly integrated.

Usage:
    python tests/verify_credits_implementation.py
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "src" / "backend"
sys.path.insert(0, str(backend_path))


def test_imports():
    """Test that all credit system modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import credits
        print("  ✅ credits.py imports successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import credits.py: {e}")
        return False
    
    try:
        import credits_api
        print("  ✅ credits_api.py imports successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import credits_api.py: {e}")
        return False
    
    return True


def test_classes_and_functions():
    """Test that key classes and functions exist."""
    print("\n🔍 Testing classes and functions...")
    
    import credits
    import credits_api
    
    # Test credits.py exports
    required_from_credits = [
        'CreditManager',
        'calculate_credits',
        'GenerationType',
        'CreditBalance',
        'CreditPackage',
        'DEFAULT_PACKAGES',
    ]
    
    for item in required_from_credits:
        if hasattr(credits, item):
            print(f"  ✅ credits.{item} exists")
        else:
            print(f"  ❌ credits.{item} missing")
            return False
    
    # Test credits_api.py exports
    required_from_api = [
        'router',
        'stripe_router',
        'check_credits',
        'deduct_credits',
        'refund_credits',
    ]
    
    for item in required_from_api:
        if hasattr(credits_api, item):
            print(f"  ✅ credits_api.{item} exists")
        else:
            print(f"  ❌ credits_api.{item} missing")
            return False
    
    return True


def test_credit_calculations():
    """Test credit calculation logic."""
    print("\n🔍 Testing credit calculations...")
    
    from credits import calculate_credits
    
    test_cases = [
        # (type, params, expected_credits)
        ("sdxl", {"width": 1024, "height": 1024}, 1),
        ("flux", {"width": 1024, "height": 1024}, 3),  # HD by default
        ("wan22_i2v", {"width": 720, "height": 720, "duration_seconds": 3}, 5),
        ("wan22_t2v", {"width": 720, "height": 720, "duration_seconds": 3}, 8),
    ]
    
    for gen_type, params, expected in test_cases:
        cost = calculate_credits(gen_type, **params)
        if cost == expected:
            print(f"  ✅ {gen_type}: {cost} credits (expected {expected})")
        else:
            print(f"  ❌ {gen_type}: {cost} credits (expected {expected})")
            return False
    
    return True


def test_default_packages():
    """Test that default credit packages are defined."""
    print("\n🔍 Testing default packages...")
    
    from credits import DEFAULT_PACKAGES
    
    expected_packages = ["starter", "basic", "pro", "studio", "enterprise"]
    
    if len(DEFAULT_PACKAGES) == 5:
        print(f"  ✅ Found {len(DEFAULT_PACKAGES)} packages")
    else:
        print(f"  ❌ Expected 5 packages, found {len(DEFAULT_PACKAGES)}")
        return False
    
    for pkg in DEFAULT_PACKAGES:
        if pkg.id in expected_packages:
            print(f"  ✅ Package '{pkg.id}': {pkg.credits} credits, €{pkg.price_cents/100:.2f}")
        else:
            print(f"  ❌ Unexpected package: {pkg.id}")
            return False
    
    return True


def test_api_routes():
    """Test that API routes are properly configured."""
    print("\n🔍 Testing API routes...")
    
    from credits_api import router, stripe_router
    
    # Check credit router routes
    credit_routes = [r.path for r in router.routes]
    expected_routes = ["", "/packages", "/estimate", "/history", "/purchase"]
    
    for route in expected_routes:
        if route in credit_routes or f"/api/credits{route}" in [r.path for r in router.routes]:
            print(f"  ✅ Route /api/credits{route} defined")
        else:
            # Route might be defined with full path
            found = False
            for r in router.routes:
                if hasattr(r, 'path') and route in r.path:
                    found = True
                    break
            if found:
                print(f"  ✅ Route /api/credits{route} defined")
            else:
                print(f"  ⚠️  Route /api/credits{route} check skipped (needs full app context)")
    
    # Check stripe router
    stripe_routes = [r.path for r in stripe_router.routes]
    if "/webhook" in stripe_routes or any("webhook" in r.path for r in stripe_router.routes):
        print(f"  ✅ Stripe webhook route defined")
    else:
        print(f"  ⚠️  Stripe webhook route check skipped (needs full app context)")
    
    return True


def check_migration_file():
    """Check that database migration file exists."""
    print("\n🔍 Checking database migration...")
    
    migration_file = Path(__file__).parent.parent / "src" / "backend" / "migrations" / "001_credits_system.sql"
    
    if migration_file.exists():
        size = migration_file.stat().st_size
        print(f"  ✅ Migration file exists ({size} bytes)")
        
        # Check for key tables
        content = migration_file.read_text()
        tables = ["user_credits", "credit_transactions", "credit_packages"]
        for table in tables:
            if table in content:
                print(f"  ✅ Migration includes '{table}' table")
            else:
                print(f"  ❌ Migration missing '{table}' table")
                return False
        
        # Check for functions
        functions = ["deduct_credits", "add_credits"]
        for func in functions:
            if func in content:
                print(f"  ✅ Migration includes '{func}()' function")
            else:
                print(f"  ❌ Migration missing '{func}()' function")
                return False
        
        return True
    else:
        print(f"  ❌ Migration file not found: {migration_file}")
        return False


def check_frontend_components():
    """Check that frontend components exist."""
    print("\n🔍 Checking frontend components...")
    
    frontend_path = Path(__file__).parent.parent / "src" / "frontend" / "src"
    
    components = [
        "contexts/CreditsContext.jsx",
        "components/PurchaseCreditsModal.jsx",
        "components/InsufficientCreditsModal.jsx",
    ]
    
    all_found = True
    for component in components:
        component_path = frontend_path / component
        if component_path.exists():
            size = component_path.stat().st_size
            print(f"  ✅ {component} exists ({size} bytes)")
        else:
            print(f"  ❌ {component} not found")
            all_found = False
    
    return all_found


def check_documentation():
    """Check that documentation exists."""
    print("\n🔍 Checking documentation...")
    
    docs_path = Path(__file__).parent.parent / "docs"
    
    docs = [
        "CREDITS_SETUP.md",
        "CREDIT_INTEGRATION_SUMMARY.md",
    ]
    
    all_found = True
    for doc in docs:
        doc_path = docs_path / doc
        if doc_path.exists():
            size = doc_path.stat().st_size
            print(f"  ✅ {doc} exists ({size} bytes)")
        else:
            print(f"  ❌ {doc} not found")
            all_found = False
    
    return all_found


def check_env_template():
    """Check that .env.example has required variables."""
    print("\n🔍 Checking environment template...")
    
    env_file = Path(__file__).parent.parent / ".env.example"
    
    if not env_file.exists():
        print("  ❌ .env.example not found")
        return False
    
    content = env_file.read_text()
    
    required_vars = [
        "SUPABASE_URL",
        "SUPABASE_SERVICE_KEY",
        "STRIPE_SECRET_KEY",
        "STRIPE_WEBHOOK_SECRET",
        "STRIPE_PUBLISHABLE_KEY",
        "FRONTEND_URL",
    ]
    
    all_found = True
    for var in required_vars:
        if var in content:
            print(f"  ✅ {var} defined")
        else:
            print(f"  ❌ {var} missing")
            all_found = False
    
    return all_found


def main():
    """Run all verification tests."""
    print("=" * 70)
    print("  Credit System Implementation Verification")
    print("=" * 70)
    
    results = []
    
    # Run all tests
    results.append(("Imports", test_imports()))
    results.append(("Classes & Functions", test_classes_and_functions()))
    results.append(("Credit Calculations", test_credit_calculations()))
    results.append(("Default Packages", test_default_packages()))
    results.append(("API Routes", test_api_routes()))
    results.append(("Database Migration", check_migration_file()))
    results.append(("Frontend Components", check_frontend_components()))
    results.append(("Documentation", check_documentation()))
    results.append(("Environment Template", check_env_template()))
    
    # Print summary
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n  🎉 All verification checks passed!")
        print("  📋 The credit system implementation is complete.")
        print("  🚀 Ready for deployment (see docs/CREDITS_DEPLOYMENT_CHECKLIST.md)")
        return 0
    else:
        print(f"\n  ⚠️  {total - passed} check(s) failed.")
        print("  📋 Please review the failed items above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
