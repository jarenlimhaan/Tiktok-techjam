import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { query } = await request.json()

    if (!query) {
      return NextResponse.json({ error: "Query parameter is required" }, { status: 400 })
    }

    const apiKey = process.env.GOOGLE_PLACES_API_KEY

    if (!apiKey) {
      return NextResponse.json({ error: "Google Places API key not configured" }, { status: 500 })
    }

    // Step 1: Search for places using Text Search
    const searchResponse = await fetch("https://places.googleapis.com/v1/places:searchText", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": apiKey,
        "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.rating,places.userRatingCount",
      },
      body: JSON.stringify({
        textQuery: query,
        maxResultCount: 1,
      }),
    })

    if (!searchResponse.ok) {
      throw new Error(`Places search failed: ${searchResponse.statusText}`)
    }

    const searchData = await searchResponse.json()

    if (!searchData.places || searchData.places.length === 0) {
      return NextResponse.json({ error: "No places found for the given query" }, { status: 404 })
    }

    const place = searchData.places[0]

    // Step 2: Get place details including reviews
    const detailsResponse = await fetch(`https://places.googleapis.com/v1/places/${place.id}`, {
      headers: {
        "X-Goog-Api-Key": apiKey,
        "X-Goog-FieldMask": "id,displayName,formattedAddress,rating,userRatingCount,reviews",
      },
    })

    if (!detailsResponse.ok) {
      throw new Error(`Place details failed: ${detailsResponse.statusText}`)
    }

    const detailsData = await detailsResponse.json()

    // Format the response to match our interface
    const formattedResponse = {
      name: detailsData.displayName?.text || "Unknown",
      formatted_address: detailsData.formattedAddress || "Address not available",
      rating: detailsData.rating || 0,
      user_ratings_total: detailsData.userRatingCount || 0,
      reviews:
        detailsData.reviews?.map((review: {
          authorAttribution?: { displayName?: string; uri?: string; photoUri?: string };
          originalText?: { languageCode?: string; text?: string };
          text?: { text?: string };
          rating?: number;
          relativePublishTimeDescription?: string;
          publishTime?: string;
        }) => ({
          author_name: review.authorAttribution?.displayName || "Anonymous",
          author_url: review.authorAttribution?.uri || "",
          language: review.originalText?.languageCode || "en",
          profile_photo_url: review.authorAttribution?.photoUri || "",
          rating: review.rating || 0,
          relative_time_description: review.relativePublishTimeDescription || "Recently",
          text: review.originalText?.text || review.text?.text || "",
          time: review.publishTime ? new Date(review.publishTime).getTime() / 1000 : Date.now() / 1000,
        })) || [],
    }

    return NextResponse.json(formattedResponse)
  } catch (error) {
    console.error("Places API error:", error)
    return NextResponse.json({ error: "Failed to fetch place data" }, { status: 500 })
  }
}
