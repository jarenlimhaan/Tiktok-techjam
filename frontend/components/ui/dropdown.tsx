import { forwardRef } from "react"
import { cn } from "@/lib/utils"

interface DropdownProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode
}

interface DropdownItemProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode
  selected?: boolean
}

const Dropdown = forwardRef<HTMLDivElement, DropdownProps>(
  ({ className, children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          "absolute z-50 w-full mt-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md shadow-lg max-h-60 overflow-auto",
          className
        )}
        {...props}
      >
        {children}
      </div>
    )
  }
)
Dropdown.displayName = "Dropdown"

const DropdownItem = forwardRef<HTMLDivElement, DropdownItemProps>(
  ({ className, children, selected, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          "px-3 py-2 cursor-pointer text-sm hover:bg-gray-100 dark:hover:bg-gray-700 border-b border-gray-100 dark:border-gray-700 last:border-b-0",
          selected && "bg-gray-100 dark:bg-gray-700",
          className
        )}
        {...props}
      >
        {children}
      </div>
    )
  }
)
DropdownItem.displayName = "DropdownItem"

export { Dropdown, DropdownItem }