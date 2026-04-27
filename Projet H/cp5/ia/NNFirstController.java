package fr.uha.hassenforder.nn;

import java.util.Map;
import java.util.TreeMap;
import javax.ws.rs.QueryParam;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

@RestController
public class NNFirstController {

    @PostMapping (value="/config")
    @ResponseStatus(HttpStatus.OK)
    public String config (@QueryParam( "graph" ) String graph) {
        
        System.err.println("Graph="+graph);
        return "ok"+graph;
    }

    @PostMapping (value="/classify")
    @ResponseStatus(HttpStatus.OK)
    public String classify (@RequestParam( "picture" ) MultipartFile picture) {
        return "ok";
    }
}
