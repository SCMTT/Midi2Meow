"""
Observer pattern implementation for m2m framework.

Enables GUI components to observe changes in the core pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Callable, Any
from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    """Types of events that can be observed."""

    PROGRESS_START = "progress_start"
    PROGRESS_UPDATE = "progress_update"
    PROGRESS_COMPLETE = "progress_complete"
    SEGMENTATION_COMPLETE = "segmentation_complete"
    TRANSPOSITION_COMPLETE = "transposition_complete"
    ERROR = "error"
    STATUS_UPDATE = "status_update"


@dataclass
class Event:
    """Event data passed to observers."""

    type: EventType
    data: Any = None
    message: str = ""


class Observer(ABC):
    """Abstract observer interface."""

    @abstractmethod
    def update(self, event: Event) -> None:
        """
        Called when observed object changes.

        Args:
            event: Event object containing type, data, and message
        """
        pass


class Observable:
    """
    Subject class that maintains a list of observers and notifies them of changes.
    """

    def __init__(self):
        self._observers: List[Observer] = []

    def attach(self, observer: Observer) -> None:
        """
        Attach an observer to this subject.

        Args:
            observer: Observer to attach
        """
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        """
        Detach an observer from this subject.

        Args:
            observer: Observer to detach
        """
        try:
            self._observers.remove(observer)
        except ValueError:
            pass

    def notify(self, event: Event) -> None:
        """
        Notify all observers of an event.

        Args:
            event: Event to notify observers about
        """
        for observer in self._observers:
            observer.update(event)

    def clear_observers(self) -> None:
        """Remove all observers."""
        self._observers.clear()


class FunctionObserver(Observer):
    """
    Observer that calls a function when updated.

    Useful for simple callbacks without creating a full Observer class.
    """

    def __init__(self, callback: Callable[[Event], None]):
        """
        Initialize with a callback function.

        Args:
            callback: Function to call when update() is invoked
        """
        self.callback = callback

    def update(self, event: Event) -> None:
        """Call the callback with the event."""
        self.callback(event)


class ProgressTracker(Observable):
    """
    Tracks progress of operations and notifies observers.

    Usage:
        tracker = ProgressTracker()
        tracker.attach(gui_progress_bar)
        tracker.start("Processing MIDI", total_steps=3)
        tracker.update(1, "Segmenting...")
        tracker.update(2, "Transposing...")
        tracker.complete()
    """

    def __init__(self):
        super().__init__()
        self._current = 0
        self._total = 0
        self._operation = ""

    def start(self, operation: str, total_steps: int = 0) -> None:
        """
        Start tracking a new operation.

        Args:
            operation: Description of the operation
            total_steps: Total number of steps (0 for indeterminate)
        """
        self._operation = operation
        self._total = total_steps
        self._current = 0

        self.notify(
            Event(
                type=EventType.PROGRESS_START,
                data={"operation": operation, "total": total_steps},
                message=operation,
            )
        )

    def update(self, step: int = None, message: str = "") -> None:
        """
        Update progress.

        Args:
            step: Current step number (optional, increments if None)
            message: Status message
        """
        if step is not None:
            self._current = step
        else:
            self._current += 1

        self.notify(
            Event(
                type=EventType.PROGRESS_UPDATE,
                data={
                    "operation": self._operation,
                    "current": self._current,
                    "total": self._total,
                    "progress": self._current / self._total if self._total > 0 else 0,
                },
                message=message or f"{self._operation}: {self._current}/{self._total}",
            )
        )

    def complete(self, message: str = "Complete") -> None:
        """
        Mark operation as complete.

        Args:
            message: Completion message
        """
        self.notify(
            Event(
                type=EventType.PROGRESS_COMPLETE,
                data={"operation": self._operation},
                message=message,
            )
        )

    def error(self, error_message: str) -> None:
        """
        Report an error.

        Args:
            error_message: Error description
        """
        self.notify(
            Event(
                type=EventType.ERROR,
                data={"operation": self._operation},
                message=error_message,
            )
        )
