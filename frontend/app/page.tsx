"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Star, MapPin, Clock, User } from "lucide-react"
import { cn } from "@/lib/utils"

interface Review {
  author_name: string
  author_url?: string
  language: string
  profile_photo_url?: string
  rating: number
  relative_time_description: string
  text: string
  time: number
}

interface PlaceDetails {
  name: string
  formatted_address: string
  rating: number
  user_ratings_total: number
  reviews: Review[]
  place_id: string
}

export default function GoogleReviewsApp() {
  const [searchQuery, setSearchQuery] = useState("")
  const [placeDetails, setPlaceDetails] = useState<PlaceDetails | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const searchPlace = async () => {
    if (!searchQuery.trim()) return

    setLoading(true)
    setError("")
    setPlaceDetails(null)

    try {
      const response = await fetch("/api/places/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: searchQuery }),
      })

      if (!response.ok) {
        throw new Error("Failed to search place")
      }

      const data = await response.json()
      setPlaceDetails(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setLoading(false)
    }
  }

  const renderStars = (rating: number) => {
    return Array.from({ length: 5 }, (_, i) => (
      <Star
        key={i}
        className={cn(
          "h-4 w-4",
          i < Math.floor(rating)
            ? "fill-yellow-400 text-yellow-400"
            : i < rating
              ? "fill-yellow-400/50 text-yellow-400"
              : "text-gray-300",
        )}
      />
    ))
  }

  return (
    <div className="min-h-screen bg-background p-4">
      <div className="mx-auto max-w-4xl space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold text-balance">Google Reviews Finder</h1>
          <p className="text-muted-foreground text-pretty">
            Search for any business or location to view their Google reviews
          </p>
        </div>

        {/* Search Form */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <MapPin className="h-5 w-5" />
              Search Location
            </CardTitle>
            <CardDescription>Enter a business name, address, or location to find reviews</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-2">
              <Input
                placeholder="e.g., Starbucks New York, 123 Main St, Restaurant near me"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && searchPlace()}
                className="flex-1"
              />
              <Button onClick={searchPlace} disabled={loading}>
                {loading ? "Searching..." : "Search"}
              </Button>
            </div>
            {error && <div className="text-sm text-destructive bg-destructive/10 p-3 rounded-md">{error}</div>}
          </CardContent>
        </Card>

        {/* Results */}
        {placeDetails && (
          <div className="space-y-6">
            {/* Place Info */}
            <Card>
              <CardHeader>
                <CardTitle className="text-xl">{placeDetails.name}</CardTitle>
                <CardDescription className="flex items-center gap-2">
                  <MapPin className="h-4 w-4" />
                  {placeDetails.formatted_address}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <div className="flex">{renderStars(placeDetails.rating)}</div>
                    <span className="font-semibold">{placeDetails.rating}</span>
                  </div>
                  <Badge variant="secondary">{placeDetails.user_ratings_total} reviews</Badge>
                </div>
              </CardContent>
            </Card>

            {/* Reviews */}
            <div className="space-y-4">
              <h2 className="text-2xl font-semibold">Reviews</h2>
              {placeDetails.reviews.length === 0 ? (
                <Card>
                  <CardContent className="py-8 text-center text-muted-foreground">
                    No reviews available for this location
                  </CardContent>
                </Card>
              ) : (
                <div className="grid gap-4">
                  {placeDetails.reviews.map((review, index) => (
                    <Card key={index}>
                      <CardContent className="pt-6">
                        <div className="space-y-3">
                          {/* Review Header */}
                          <div className="flex items-start justify-between">
                            <div className="flex items-center gap-3">
                              {review.profile_photo_url ? (
                                <img
                                  src={review.profile_photo_url || "/placeholder.svg"}
                                  alt={review.author_name}
                                  className="h-10 w-10 rounded-full"
                                />
                              ) : (
                                <div className="h-10 w-10 rounded-full bg-muted flex items-center justify-center">
                                  <User className="h-5 w-5" />
                                </div>
                              )}
                              <div>
                                <p className="font-semibold">{review.author_name}</p>
                                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                  <Clock className="h-3 w-3" />
                                  {review.relative_time_description}
                                </div>
                              </div>
                            </div>
                            <div className="flex items-center gap-1">{renderStars(review.rating)}</div>
                          </div>

                          {/* Review Text */}
                          {review.text && <p className="text-sm leading-relaxed text-pretty">{review.text}</p>}
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* API Notice */}
        <Card className="border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-950">
          <CardContent className="pt-6">
            <div className="space-y-2">
              <h3 className="font-semibold text-blue-900 dark:text-blue-100">🔑 API Setup Required</h3>
              <p className="text-sm text-blue-800 dark:text-blue-200">
                This app uses the official Google Places API. You'll need to:
              </p>
              <ul className="text-sm text-blue-800 dark:text-blue-200 list-disc list-inside space-y-1">
                <li>Get a Google Places API key from Google Cloud Console</li>
                <li>Enable the Places API (New) service</li>
                <li>Add your API key to environment variables as GOOGLE_PLACES_API_KEY</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
