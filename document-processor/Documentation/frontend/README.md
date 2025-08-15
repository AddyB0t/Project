# Centria Frontend - Flutter Project Management Application

## Overview

Centria is a comprehensive project management Flutter application that provides multi-platform support for web, iOS, and Android. The application combines traditional project management features with modern AI-powered document processing capabilities, real-time collaboration tools, and advanced multimedia handling.

## 🎯 Key Features

### Core Project Management
- **Project Creation & Management**: Complete project lifecycle management with creation, editing, and tracking
- **Task Management**: Advanced task assignment, progress tracking, and checklist functionality
- **Workspace Organization**: Multi-workspace support with role-based access control
- **Team Collaboration**: Real-time chat, group discussions, and team member management
- **Meeting Integration**: Seamless integration with Zoom and Microsoft Teams for scheduling and management

### AI & Document Processing
- **AI Chat Assistant**: Modern AI-powered chat interface for project assistance and document queries
- **Document Management**: Upload, organize, and process various document formats (PDF, DOCX, images)
- **Image Processing**: Advanced image handling with cropping, editing, and gallery management
- **File Management**: Comprehensive file handling with preview capabilities

### Communication & Collaboration
- **Real-time Chat**: WebSocket-based instant messaging with group and direct chat support
- **Meeting Scheduler**: Integrated calendar with timezone support and external platform integration
- **Notification System**: Push notifications and in-app notification management
- **Activity Tracking**: Project activity feeds and progress updates

### Multimedia & Location
- **Camera Integration**: In-app camera functionality with geo-tagging capabilities
- **Photo Gallery**: Advanced photo management with slideshow and editing features
- **Map Integration**: Google Maps and Mapbox integration for location-based features
- **Media Playback**: Support for video and audio file playback

### Business Features
- **Payment Processing**: Integrated payment methods including PayPal, Google Pay, and Apple Pay
- **Subscription Management**: Tiered subscription plans with feature access control
- **Multi-language Support**: Internationalization with English and Spanish language support
- **Offline Capability**: Offline data synchronization and caching

## 🏗️ Architecture

### Technology Stack
- **Framework**: Flutter 3.0.6+ with Dart SDK
- **State Management**: Riverpod for reactive state management
- **Navigation**: GoRouter for declarative routing
- **HTTP Client**: Dio for network requests with interceptors
- **UI Framework**: Material Design with custom styling

### Key Dependencies
```yaml
- flutter_riverpod: ^2.4.0      # State management
- go_router: ^15.1.2            # Navigation
- dio: ^5.8.0+1                 # HTTP client
- firebase_core: ^3.13.1        # Firebase integration
- google_maps_flutter: ^2.7.0   # Maps integration
- webview_flutter: ^4.13.0      # In-app browser
- flutter_chat_ui: ^1.6.12      # Chat interface
- camera: ^0.11.1               # Camera functionality
```

### Project Structure
```
lib/
├── assets/                     # Static assets (images, SVGs, icons)
├── controllers/               # Business logic controllers
├── core/                      # Core utilities and constants
├── domain/                    # Domain models and use cases
├── infrastructure/            # External service implementations
├── layout/                    # Responsive layout components
├── models/                    # Data models and state classes
├── pages/                     # Main application pages
├── presentation/              # UI components and view models
├── screens/                   # Screen implementations
├── services/                  # Service layer implementations
├── widgets/                   # Reusable UI components
├── main.dart                  # Application entry point
└── routes.dart                # Route definitions
```

## 🔧 Configuration

### Environment Setup
```bash
# Install dependencies
flutter pub get

# Generate localizations
flutter gen-l10n

# Run code generation
flutter packages pub run build_runner build
```

### Platform Configuration

#### Web Deployment
```bash
# Build for web
flutter build web --release --web-renderer html

# Serve locally
node server_local.js
```

#### Mobile Deployment
- **Android**: Configure `android/app/build.gradle` and Google Services
- **iOS**: Configure `ios/Runner/Info.plist` and Apple Developer settings

### Firebase Configuration
- Firebase Core for cross-platform initialization
- Firebase Auth for authentication (handled by backend)
- Firebase Crashlytics for error reporting
- Firebase Analytics for usage tracking

## 📱 Screen Architecture

### Authentication Flow
- **Login Screen**: OAuth integration with Google Sign-In
- **Registration**: Multi-step user onboarding
- **Password Recovery**: Email-based password reset with OTP verification

### Main Application Screens
- **Home Dashboard**: Project overview with customizable modules
- **Project Management**: Create, edit, and manage projects
- **Task Management**: Task assignment and progress tracking
- **Chat Interface**: Real-time messaging with multimedia support
- **Meeting Management**: Schedule and join meetings
- **Document Viewer**: PDF reader with annotation capabilities
- **Settings**: User preferences and account management

### Navigation Structure
The application uses a sophisticated routing system with 40+ defined routes:
- Authentication routes (`/login`, `/sign-up`, `/forgot`)
- Main application routes (`/home`, `/projects`, `/chat`)
- Settings and profile routes (`/settings`, `/profile`)
- Feature-specific routes (`/ai-chat`, `/pdf-reader`, `/gallery`)

## 🎨 UI/UX Design

### Design System
- **Color Scheme**: Custom brown and blue theme with light/dark mode support
- **Typography**: Google Fonts integration with consistent text styling
- **Icons**: Material Icons, FontAwesome, and custom SVG icons
- **Responsive Design**: Adaptive layouts for web, tablet, and mobile

### Key UI Components
- Custom chat bubbles with rich message types
- Interactive project cards with progress indicators
- Advanced photo gallery with zoom and editing capabilities
- Customizable dashboard modules
- Material Design components with custom styling

## 🔌 Integration Points

### Backend API Integration
- RESTful API communication via Dio HTTP client
- Real-time WebSocket connections for chat and notifications
- File upload handling with progress tracking
- Error handling and retry mechanisms

### External Services
- **Google Services**: Maps, Sign-In, and location services
- **Video Conferencing**: Zoom and Microsoft Teams SDK integration
- **Payment Processing**: PayPal, Apple Pay, and Google Pay
- **Cloud Storage**: Firebase Storage for media files

### AI Features
- Modern AI chat interface with context-aware responses
- Document processing integration with backend AI services
- Image analysis and processing capabilities

## 🚀 Deployment

### Web Deployment
- Build output directory: `build/web`
- Vercel deployment configuration included
- Static web app configuration for Azure

### Mobile App Stores
- **Google Play Store**: Android app bundle configuration
- **Apple App Store**: iOS archive and distribution setup

### Development Workflow
```bash
# Development server
flutter run -d chrome  # Web
flutter run -d ios     # iOS Simulator
flutter run -d android # Android Emulator

# Production builds
flutter build web --release
flutter build apk --release
flutter build ios --release
```

## 📊 Performance & Analytics

### Performance Optimization
- Image caching with `flutter_cache_manager`
- Lazy loading for large lists and galleries
- Offline data synchronization
- Memory-efficient state management

### Analytics & Monitoring
- Firebase Analytics for user behavior tracking
- Crashlytics for error reporting and performance monitoring
- Custom event tracking for feature usage

## 🔒 Security Features

### Data Protection
- Secure storage for sensitive data using `flutter_secure_storage`
- Encrypted local data persistence
- Secure network communication with SSL/TLS

### Privacy Compliance
- GDPR-compliant data handling
- User consent management
- Privacy policy integration

## 🧪 Testing

### Testing Strategy
- Unit tests for business logic and models
- Widget tests for UI components
- Integration tests for complete user flows
- Mock testing with `mocktail` package

### Test Structure
```
test/
├── presentation/
│   └── view_model/
│       └── project_provider_test.dart
└── widget_test.dart
```

## 📚 Development Guidelines

### Code Style
- Follow Flutter/Dart style guidelines
- Use meaningful variable and function names
- Implement proper error handling
- Document complex business logic

### State Management Patterns
- Use Riverpod providers for application state
- Implement proper dependency injection
- Separate business logic from UI components
- Use immutable data models

## 🌐 Internationalization

### Supported Languages
- English (en)
- Spanish (es)

### Localization Files
```
l10n/
├── app_en.arb
└── app_es.arb
```

## 🔮 Future Enhancements

### Planned Features
- Enhanced AI capabilities for project insights
- Advanced reporting and analytics dashboards
- Additional language support
- Improved offline functionality
- Advanced document collaboration features

---

## 📞 Support & Documentation

For technical support or questions about the frontend implementation:
- Review the source code documentation
- Check the Flutter official documentation
- Consult the backend API documentation for integration details

**Version**: 1.0.3+4  
**Last Updated**: August 2025  
**Minimum Flutter Version**: 3.0.6