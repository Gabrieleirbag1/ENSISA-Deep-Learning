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
public class NNController {

    @PostMapping (value="/config")
    @ResponseStatus(HttpStatus.OK)
    public String config (@QueryParam( "graph" ) String graph) {
        
        System.err.println("Graph="+graph);
        String url = "http://127.0.0.1:8081/config?graph={graph}";

        HttpHeaders headers = new HttpHeaders();
        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(headers);
        
        Map<String, String> params = new TreeMap<>();
        params.put("graph", graph);

        RestTemplate template = new RestTemplate();
        ResponseEntity<String> response = template.exchange(url, HttpMethod.POST, requestEntity, String.class, params);

        return response.getBody();
    }

    @PostMapping (value="/classify")
    @ResponseStatus(HttpStatus.OK)
    public String classify (@RequestParam( "picture" ) MultipartFile picture) {
        String url = "http://127.0.0.1:8081/classify";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("picture", picture.getResource());
        
        Map<String, String> params = new TreeMap<>();

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
        
        RestTemplate template = new RestTemplate();
        ResponseEntity<String> response = template.exchange(url, HttpMethod.POST, requestEntity, String.class, params);

        return response.getBody();
    }
}
