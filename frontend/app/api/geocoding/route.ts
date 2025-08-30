import { type NextRequest, NextResponse } from "next/server"

interface MapboxFeature {
  id: string
  type: string
  place_name: string
  properties: {
    address?: string
    category?: string
  }
  center: [number, number]
  place_type: string[]
  context?: Array<{
    id: string
    text: string
  }>
}

interface MapboxGeocodingResponse {
  type: string
  query: string[]
  features: MapboxFeature[]
}

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const query = searchParams.get("q")

    if (!query) {
      return NextResponse.json({ error: "Query parameter 'q' is required" }, { status: 400 })
    }

    const apiKey = process.env.MAPBOX_ACCESS_TOKEN

    if (!apiKey) {
      return NextResponse.json({ error: "Mapbox access token not configured" }, { status: 500 })
    }

    // Use Mapbox Geocoding API
    const mapboxUrl = `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(query)}.json`
    const params = new URLSearchParams({
      access_token: apiKey,
      limit: "5", // Limit to 5 results for dropdown
      types: "poi,address,place", // Points of interest, addresses, and places
    })

    const response = await fetch(`${mapboxUrl}?${params}`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    })

    if (!response.ok) {
      throw new Error(`Mapbox geocoding failed: ${response.statusText}`)
    }

    const data: MapboxGeocodingResponse = await response.json()

    // Format the response for the frontend dropdown
    const formattedResults = data.features.map((feature) => ({
      id: feature.id,
      name: feature.place_name,
      address: feature.properties?.address || feature.place_name,
      coordinates: {
        latitude: feature.center[1],
        longitude: feature.center[0],
      },
      place_type: feature.place_type,
    }))

    return NextResponse.json({
      results: formattedResults,
      query: query,
    })
  } catch (error) {
    console.error("Mapbox geocoding error:", error)
    return NextResponse.json({ error: "Failed to search locations" }, { status: 500 })
  }
}