"use client"

import { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Dropdown, DropdownItem } from "@/components/ui/dropdown"
import { Star, MapPin, Clock, User, Search } from "lucide-react"
import { cn } from "@/lib/utils"
import { useDebounce } from "@/hooks/useDebounce"

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

interface GeocodingResult {
  id: string
  name: string
  address: string
  coordinates: {
    latitude: number
    longitude: number
  }
  place_type: string[]
}

export default function GoogleReviewsApp() {
  const [searchQuery, setSearchQuery] = useState("")
  const [placeDetails, setPlaceDetails] = useState<PlaceDetails | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [searchResults, setSearchResults] = useState<GeocodingResult[]>([])
  const [showDropdown, setShowDropdown] = useState(false)
  const [selectedIndex, setSelectedIndex] = useState(-1)
  const [searchLoading, setSearchLoading] = useState(false)
  
  const searchInputRef = useRef<HTMLInputElement>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)
  
  // Debounce search query by 1 second
  const debouncedSearchQuery = useDebounce(searchQuery, 1000)

  // Effect to handle debounced search
  useEffect(() => {
    if (debouncedSearchQuery.trim() && debouncedSearchQuery.length > 2) {
      searchLocations(debouncedSearchQuery)
    } else {
      setSearchResults([])
      setShowDropdown(false)
    }
  }, [debouncedSearchQuery])

  // Effect to handle outside clicks
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node) &&
        searchInputRef.current &&
        !searchInputRef.current.contains(event.target as Node)
      ) {
        setShowDropdown(false)
        setSelectedIndex(-1)
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [])

  const searchLocations = async (query: string) => {
    setSearchLoading(true)
    try {
      const response = await fetch(`/api/geocoding?q=${encodeURIComponent(query)}`)
      
      if (!response.ok) {
        throw new Error("Failed to search locations")
      }

      const data = await response.json()
      setSearchResults(data.results || [])
      setShowDropdown(data.results && data.results.length > 0)
      setSelectedIndex(-1)
    } catch (err) {
      console.error("Search error:", err)
      setSearchResults([])
      setShowDropdown(false)
    } finally {
      setSearchLoading(false)
    }
  }

  const selectLocation = async (result: GeocodingResult) => {
    setSearchQuery(result.name)
    setShowDropdown(false)
    setSelectedIndex(-1)
    
    // Use the selected location name to search for place details
    await searchPlace(result.name)
  }

  const searchPlace = async (query?: string) => {
    const searchTerm = query || searchQuery
    if (!searchTerm.trim()) return

    setLoading(true)
    setError("")
    setPlaceDetails(null)

    try {
      const response = await fetch("/api/places/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: searchTerm }),
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

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!showDropdown || searchResults.length === 0) return

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault()
        setSelectedIndex(prev => 
          prev < searchResults.length - 1 ? prev + 1 : prev
        )
        break
      case "ArrowUp":
        e.preventDefault()
        setSelectedIndex(prev => prev > 0 ? prev - 1 : -1)
        break
      case "Enter":
        e.preventDefault()
        if (selectedIndex >= 0 && selectedIndex < searchResults.length) {
          selectLocation(searchResults[selectedIndex])
        } else if (searchQuery.trim()) {
          setShowDropdown(false)
          searchPlace()
        }
        break
      case "Escape":
        setShowDropdown(false)
        setSelectedIndex(-1)
        break
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
          <h1 className="text-3xl font-bold text-balance">Location Review Finder</h1>
          <p className="text-muted-foreground text-pretty">
            Search for any business or location to view their reviews using Mapbox geocoding
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
            <div className="relative">
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <Input
                    ref={searchInputRef}
                    placeholder="e.g., Starbucks New York, 123 Main St, Restaurant near me"
                    value={searchQuery}
                    onChange={(e) => {
                      setSearchQuery(e.target.value)
                      if (e.target.value.trim()) {
                        setShowDropdown(true)
                      }
                    }}
                    onKeyDown={handleKeyDown}
                    onFocus={() => {
                      if (searchResults.length > 0) {
                        setShowDropdown(true)
                      }
                    }}
                    className="pr-10"
                  />
                  {searchLoading && (
                    <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                      <Search className="h-4 w-4 animate-spin" />
                    </div>
                  )}
                  
                  {/* Search Results Dropdown */}
                  {showDropdown && searchResults.length > 0 && (
                    <Dropdown ref={dropdownRef}>
                      {searchResults.map((result, index) => (
                        <DropdownItem
                          key={result.id}
                          selected={index === selectedIndex}
                          onClick={() => selectLocation(result)}
                          className="space-y-1"
                        >
                          <div className="font-medium text-sm">{result.name}</div>
                          <div className="text-xs text-muted-foreground">{result.address}</div>
                          <div className="flex gap-1">
                            {result.place_type.map((type) => (
                              <Badge key={type} variant="secondary" className="text-xs px-1 py-0">
                                {type}
                              </Badge>
                            ))}
                          </div>
                        </DropdownItem>
                      ))}
                    </Dropdown>
                  )}
                </div>
                <Button onClick={() => searchPlace()} disabled={loading}>
                  {loading ? "Searching..." : "Search"}
                </Button>
              </div>
              {error && <div className="text-sm text-destructive bg-destructive/10 p-3 rounded-md mt-2">{error}</div>}
            </div>
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
                This app uses the Mapbox Geocoding API for location search and Google Places API for reviews. You&apos;ll need to:
              </p>
              <ul className="text-sm text-blue-800 dark:text-blue-200 list-disc list-inside space-y-1">
                <li>Get a Mapbox access token from Mapbox account dashboard</li>
                <li>Get a Google Places API key from Google Cloud Console</li>
                <li>Enable the Places API (New) service in Google Cloud</li>
                <li>Add your Mapbox token to environment variables as MAPBOX_ACCESS_TOKEN</li>
                <li>Add your Google API key to environment variables as GOOGLE_PLACES_API_KEY</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
