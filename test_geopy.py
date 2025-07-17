#!/usr/bin/env python3
"""
Test geopy installation and functionality
"""

try:
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    print("✅ Geopy imported successfully!")
    
    # Test basic functionality
    geolocator = Nominatim(user_agent="test_app")
    
    # Test Bern coordinates
    bern = (46.9479, 7.4474)
    print(f"✅ Bern coordinates: {bern}")
    
    # Test geocoding
    location = geolocator.geocode("Zurich, Switzerland", timeout=10)
    if location:
        print(f"✅ Geocoding test successful: Zurich at {location.latitude}, {location.longitude}")
        
        # Test distance calculation
        zurich = (location.latitude, location.longitude)
        distance = geodesic(bern, zurich).kilometers
        print(f"✅ Distance calculation successful: Bern to Zurich = {distance:.1f} km")
    else:
        print("❌ Geocoding test failed")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")
